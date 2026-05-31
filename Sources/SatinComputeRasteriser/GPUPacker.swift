#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd

/// GPU implementation of ``PackedPointCloudFixtures/pack(positions:colors:pointsPerBatch:lodLevels:coarseVoxelDivisions:)``.
///
/// Fills a ``ComputeRasteriserPointCloud``'s slot pool directly from
/// GPU-resident `positions`/`colors` buffers — no CPU round-trip, no Swift
/// arrays. Intended for callers whose point data already lives in
/// `.storageModeShared` `MTLBuffer`s (e.g. a geometry-node engine that
/// evaluates on the GPU) and that re-realize a render cloud on every edit.
///
/// The output is **decode-identical** to the CPU `pack()` (same Morton order
/// up to equal-key tie order, same per-batch quantization, same LOD levels),
/// so the rasteriser's depth/color passes render it unchanged.
///
/// Stages (each a compute kernel): global bounds → Morton key → LSD radix sort
/// → gather → per-batch AABB → quantize → LOD voxel occupancy.
///
/// - Note: Apple-platform only. The CPU `pack()` remains the cross-platform
///   reference/fallback.
public final class GPUPacker {
    /// Per-tile element count for the radix sort. One thread processes a whole
    /// tile (histogram + stable scatter in identical order), so this trades
    /// per-thread serial work against tile count / scan size.
    public static let radixTileSize = 1024

    /// Maximum dense LOD-grid cell count before GPU LOD is skipped (levels left
    /// at the finest). For the default config the grid is ≤ 258³ ≈ 17M cells.
    public static let lodCellCap = 268_435_456 // 256³·16 ≈ 1 GB of uint cells

    private struct PackParams {
        var count: UInt32 = 0
        var pointsPerBatch: UInt32 = 0
        var numBatches: UInt32 = 0
        var numTiles: UInt32 = 0
        var tileSize: UInt32 = 0
        var shift: UInt32 = 0
        var maxLevel: UInt32 = 0
        var level: UInt32 = 0
        var coarseVoxelDivisions: Float = 0
        var lodVoxelScale: Float = 0
    }

    private let context: Context
    private let lodLevels: Int
    private let coarseVoxelDivisions: Int

    private let pBoundsInit, pBounds, pMorton: MTLComputePipelineState
    private let pHistogram, pScanPerDigit, pDigitBase, pScatter: MTLComputePipelineState
    private let pGather, pBatchAABB, pQuantize: MTLComputePipelineState
    private let pLODInit, pLODClaim, pLODAssign: MTLComputePipelineState

    // Scratch, grown to fit the largest pack seen so far.
    private var capacityPoints = 0
    private var keysA, keysB, indicesA, indicesB: MTLBuffer?
    private var sortedPos, boundsBuf: MTLBuffer?
    private var histBuf, tileOffsetBuf, digitTotalBuf, digitBaseBuf: MTLBuffer?
    private var lodGrid: MTLBuffer?
    private var lodGridCells = 0

    /// - Parameters:
    ///   - context: Satin context (provides the Metal device).
    ///   - lodLevels: number of LOD levels (clamped to `1...8`); matches
    ///     ``PackedPointCloudFixtures/defaultLODLevels``.
    ///   - coarseVoxelDivisions: coarsest-level voxel divisions along the
    ///     longest axis; matches ``PackedPointCloudFixtures/defaultCoarseVoxelDivisions``.
    public init(
        context: Context,
        lodLevels: Int = PackedPointCloudFixtures.defaultLODLevels,
        coarseVoxelDivisions: Int = PackedPointCloudFixtures.defaultCoarseVoxelDivisions
    ) throws {
        self.context = context
        self.lodLevels = max(1, min(lodLevels, 8))
        self.coarseVoxelDivisions = max(1, coarseVoxelDivisions)

        let url = ComputeRasteriser.pipelinesURL
            .appendingPathComponent("GPUPacker")
            .appendingPathComponent("Shaders.metal")
        let source = try String(contentsOf: url, encoding: .utf8)
        let library = try context.device.makeLibrary(source: source, options: nil)

        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw GPUPackerError.kernelNotFound(name)
            }
            return try context.device.makeComputePipelineState(function: fn)
        }
        pBoundsInit = try pipeline("packBoundsInit")
        pBounds = try pipeline("packBounds")
        pMorton = try pipeline("packMorton")
        pHistogram = try pipeline("radixHistogram")
        pScanPerDigit = try pipeline("radixScanPerDigit")
        pDigitBase = try pipeline("radixDigitBase")
        pScatter = try pipeline("radixScatter")
        pGather = try pipeline("packGather")
        pBatchAABB = try pipeline("packBatchAABB")
        pQuantize = try pipeline("packQuantize")
        pLODInit = try pipeline("packLODInit")
        pLODClaim = try pipeline("packLODClaim")
        pLODAssign = try pipeline("packLODAssign")
    }

    /// Pack GPU-resident point data into `cloud`, encoding all stages onto
    /// `commandBuffer`. The pool is grown to fit; the CPU batch mirror is
    /// re-synced from the GPU-written bounds via a completion handler.
    ///
    /// - Parameters:
    ///   - positions: `float3` positions, **16-byte stride** (`SIMD3<Float>`),
    ///     `.storageModeShared`. Caller-owned; not freed.
    ///   - colors: `float4` RGBA in `[0,1]`, 16-byte stride, `.storageModeShared`.
    ///   - count: number of points to read from `positions`/`colors`.
    ///   - cloud: destination; its pool buffers are overwritten wholesale.
    ///   - commandBuffer: the caller commits this; stages run in order.
    public func pack(
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int,
        into cloud: ComputeRasteriserPointCloud,
        commandBuffer: MTLCommandBuffer
    ) {
        guard count > 0 else {
            cloud.clearAllBatches()
            return
        }
        let ppb = cloud.capacity.pointsPerBatch
        let numBatches = cloud.prepareForGPUPack(pointCount: count)
        ensureScratch(count: count)

        guard let keysA, let keysB, let indicesA, let indicesB,
              let sortedPos, let boundsBuf,
              let histBuf, let tileOffsetBuf, let digitTotalBuf, let digitBaseBuf,
              let xyzLow = cloud.xyzLowBuffer, let xyzMed = cloud.xyzMedBuffer,
              let xyzHigh = cloud.xyzHighBuffer, let colorsOut = cloud.colorsBuffer,
              let levels = cloud.levelsBuffer, let batches = cloud.batchesBuffer
        else { return }

        let numTiles = (count + Self.radixTileSize - 1) / Self.radixTileSize
        var params = PackParams(
            count: UInt32(count),
            pointsPerBatch: UInt32(ppb),
            numBatches: UInt32(numBatches),
            numTiles: UInt32(numTiles),
            tileSize: UInt32(Self.radixTileSize),
            shift: 0,
            maxLevel: UInt32(lodLevels - 1),
            level: 0,
            coarseVoxelDivisions: Float(coarseVoxelDivisions),
            lodVoxelScale: 1.0
        )

        // 1. Global bounds.
        encode(commandBuffer, pBoundsInit, threads: 6) { e in
            e.setBuffer(boundsBuf, offset: 0, index: 0)
        }
        encodeGroups(commandBuffer, pBounds, groups: min(numTiles, 2048), threadsPerGroup: 256) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(boundsBuf, offset: 0, index: 1)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
        }

        // 2. Morton key + identity indices.
        encode(commandBuffer, pMorton, threads: count) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(keysA, offset: 0, index: 1)
            e.setBuffer(indicesA, offset: 0, index: 2)
            e.setBuffer(boundsBuf, offset: 0, index: 3)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
        }

        // 3. LSD radix sort — 4×8-bit passes, ping-ponging A→B→A→B→A.
        var (kIn, kOut, iIn, iOut) = (keysA, keysB, indicesA, indicesB)
        for pass in 0 ..< 4 {
            params.shift = UInt32(pass * 8)
            encode(commandBuffer, pHistogram, threads: numTiles) { e in
                e.setBuffer(kIn, offset: 0, index: 0)
                e.setBuffer(histBuf, offset: 0, index: 1)
                e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
            }
            encodeGroups(commandBuffer, pScanPerDigit, groups: 256, threadsPerGroup: 256) { e in
                e.setBuffer(histBuf, offset: 0, index: 0)
                e.setBuffer(tileOffsetBuf, offset: 0, index: 1)
                e.setBuffer(digitTotalBuf, offset: 0, index: 2)
                e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 3)
            }
            encode(commandBuffer, pDigitBase, threads: 1) { e in
                e.setBuffer(digitTotalBuf, offset: 0, index: 0)
                e.setBuffer(digitBaseBuf, offset: 0, index: 1)
            }
            encode(commandBuffer, pScatter, threads: numTiles) { e in
                e.setBuffer(kIn, offset: 0, index: 0)
                e.setBuffer(iIn, offset: 0, index: 1)
                e.setBuffer(kOut, offset: 0, index: 2)
                e.setBuffer(iOut, offset: 0, index: 3)
                e.setBuffer(tileOffsetBuf, offset: 0, index: 4)
                e.setBuffer(digitBaseBuf, offset: 0, index: 5)
                e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 6)
            }
            swap(&kIn, &kOut)
            swap(&iIn, &iOut)
        }
        // After 4 swaps the sorted indices are back in indicesA (== iIn).
        let sortedIndices = iIn

        // 4. Gather sorted positions + pack colors into the pool.
        encode(commandBuffer, pGather, threads: count) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(colors, offset: 0, index: 1)
            e.setBuffer(sortedIndices, offset: 0, index: 2)
            e.setBuffer(sortedPos, offset: 0, index: 3)
            e.setBuffer(colorsOut, offset: 0, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }

        // 5. Per-batch AABB → batch metadata.
        encodeGroups(commandBuffer, pBatchAABB, groups: numBatches, threadsPerGroup: 256) { e in
            e.setBuffer(sortedPos, offset: 0, index: 0)
            e.setBuffer(batches, offset: 0, index: 1)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
        }

        // 6. Quantize → xyzLow/Med/High.
        encode(commandBuffer, pQuantize, threads: count) { e in
            e.setBuffer(sortedPos, offset: 0, index: 0)
            e.setBuffer(batches, offset: 0, index: 1)
            e.setBuffer(xyzLow, offset: 0, index: 2)
            e.setBuffer(xyzMed, offset: 0, index: 3)
            e.setBuffer(xyzHigh, offset: 0, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }

        // 7. LOD voxel occupancy.
        encode(commandBuffer, pLODInit, threads: count) { e in
            e.setBuffer(levels, offset: 0, index: 0)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 1)
        }
        if lodLevels > 1, let lodGrid {
            for level in 0 ..< (lodLevels - 1) {
                params.level = UInt32(level)
                params.lodVoxelScale = powf(0.5, Float(level))
                if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.label = "GPUPacker.LODGridFill"
                    blit.fill(buffer: lodGrid, range: 0 ..< (lodGridCells * 4), value: 0xff)
                    blit.endEncoding()
                }
                encode(commandBuffer, pLODClaim, threads: count) { e in
                    e.setBuffer(sortedPos, offset: 0, index: 0)
                    e.setBuffer(levels, offset: 0, index: 1)
                    e.setBuffer(lodGrid, offset: 0, index: 2)
                    e.setBuffer(boundsBuf, offset: 0, index: 3)
                    e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
                }
                encode(commandBuffer, pLODAssign, threads: count) { e in
                    e.setBuffer(sortedPos, offset: 0, index: 0)
                    e.setBuffer(levels, offset: 0, index: 1)
                    e.setBuffer(lodGrid, offset: 0, index: 2)
                    e.setBuffer(boundsBuf, offset: 0, index: 3)
                    e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
                }
            }
        }

        commandBuffer.addCompletedHandler { _ in
            cloud.adoptGPUBatchBounds(numBatches: numBatches)
        }
    }

    // MARK: - Scratch

    private func ensureScratch(count: Int) {
        if count > capacityPoints {
            let device = context.device
            let uintLen = count * MemoryLayout<UInt32>.stride
            let numTiles = (count + Self.radixTileSize - 1) / Self.radixTileSize
            keysA = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            keysB = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            indicesA = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            indicesB = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            sortedPos = device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
            histBuf = device.makeBuffer(length: numTiles * 256 * 4, options: .storageModePrivate)
            tileOffsetBuf = device.makeBuffer(length: numTiles * 256 * 4, options: .storageModePrivate)
            keysA?.label = "GPUPacker.keysA"; keysB?.label = "GPUPacker.keysB"
            indicesA?.label = "GPUPacker.indicesA"; indicesB?.label = "GPUPacker.indicesB"
            sortedPos?.label = "GPUPacker.sortedPos"
            capacityPoints = count
        }
        if boundsBuf == nil {
            boundsBuf = context.device.makeBuffer(length: 6 * 4, options: .storageModePrivate)
            digitTotalBuf = context.device.makeBuffer(length: 256 * 4, options: .storageModePrivate)
            digitBaseBuf = context.device.makeBuffer(length: 256 * 4, options: .storageModePrivate)
            boundsBuf?.label = "GPUPacker.bounds"
        }
        ensureLODGrid()
    }

    private func ensureLODGrid() {
        guard lodLevels > 1, lodGrid == nil else { return }
        // Per-axis cell count ≤ coarseVoxelDivisions·2^(lodLevels-2) (voxel size
        // scales with the longest axis). +2 guards the floor()+1 dim formula.
        let maxDim = coarseVoxelDivisions * (1 << (lodLevels - 2)) + 2
        let cells = maxDim * maxDim * maxDim
        guard cells <= Self.lodCellCap else {
            print("[GPUPacker] LOD grid \(cells) cells exceeds cap \(Self.lodCellCap); GPU LOD disabled (levels left at finest).")
            return
        }
        lodGrid = context.device.makeBuffer(length: cells * 4, options: .storageModePrivate)
        lodGrid?.label = "GPUPacker.lodGrid"
        lodGridCells = cells
    }

    // MARK: - Encode helpers

    private func encode(
        _ commandBuffer: MTLCommandBuffer,
        _ pipeline: MTLComputePipelineState,
        threads: Int,
        _ bind: (MTLComputeCommandEncoder) -> Void
    ) {
        guard threads > 0, let e = commandBuffer.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipeline)
        bind(e)
        let tew = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tew, height: 1, depth: 1))
        e.endEncoding()
    }

    private func encodeGroups(
        _ commandBuffer: MTLCommandBuffer,
        _ pipeline: MTLComputePipelineState,
        groups: Int,
        threadsPerGroup: Int,
        _ bind: (MTLComputeCommandEncoder) -> Void
    ) {
        guard groups > 0, let e = commandBuffer.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipeline)
        bind(e)
        let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, threadsPerGroup)
        e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        e.endEncoding()
    }
}

public enum GPUPackerError: LocalizedError {
    case kernelNotFound(String)
    public var errorDescription: String? {
        switch self {
        case let .kernelNotFound(name): return "GPUPacker kernel '\(name)' not found."
        }
    }
}

extension ComputeRasteriserPointCloud {
    /// Wholesale-replace this cloud's contents from GPU-resident buffers using
    /// `packer`, on a fresh command buffer that is committed and waited on
    /// before returning. Synchronous convenience around
    /// ``GPUPacker/pack(positions:colors:count:into:commandBuffer:)``.
    ///
    /// - Parameters:
    ///   - packer: a configured ``GPUPacker``.
    ///   - queue: command queue to submit on.
    ///   - positions: `float3` positions buffer (16-byte stride, shared).
    ///   - colors: `float4` color buffer (shared).
    ///   - count: number of points.
    public func replacePackedPointCloud(
        packer: GPUPacker,
        queue: MTLCommandQueue,
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int
    ) {
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        commandBuffer.label = "GPUPacker.replacePackedPointCloud"
        packer.pack(positions: positions, colors: colors, count: count, into: self, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
#endif
