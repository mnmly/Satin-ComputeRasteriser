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
/// up to equal-key tie order, same per-batch quantization, same LOD levels,
/// same stable level-ascending bucketing with the cumulative counts of
/// ``RasterBatch/lodCumulativeCounts`` filled in), so the rasteriser's
/// depth/color passes render it unchanged.
///
/// Stages (each a compute kernel): global bounds → Morton key → LSD radix sort
/// → gather → LOD voxel occupancy → per-batch AABB → per-batch level bucketing
/// → bucketed re-gather → quantize.
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
        var slotBase: UInt32 = 0
    }

    private let context: Context
    private let lodLevels: Int
    private let coarseVoxelDivisions: Int

    private let pBoundsInit, pBounds, pMorton: MTLComputePipelineState
    private let pHistogram, pScanPerDigit, pDigitBase, pScatter: MTLComputePipelineState
    private let pGather, pBatchAABB, pQuantize: MTLComputePipelineState
    private let pLODInit, pLODClaim, pLODAssign, pBucketLOD: MTLComputePipelineState

    /// Independent scratch sets. A single set is reused (Metal hazard-tracking
    /// serialises packs that share it). ``addRawChunksGPU`` rings over several
    /// so chunks in one command buffer pack into *different* scratch and can
    /// overlap on the GPU. Each set grows to the largest pack it has seen.
    private var scratchSets: [PackScratch] = []
    /// Dense-LOD grid cell count (0 = LOD disabled: grid would exceed the cap).
    /// Computed once; each scratch set allocates its own grid of this size.
    private let lodGridCellCount: Int

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
        let clampedLevels = max(1, min(lodLevels, 8))
        let clampedDivs = max(1, coarseVoxelDivisions)
        self.lodLevels = clampedLevels
        self.coarseVoxelDivisions = clampedDivs

        // Per-axis cell count ≤ coarseVoxelDivisions·2^(lodLevels-2) (voxel size
        // scales with the longest axis). +2 guards the floor()+1 dim formula.
        if clampedLevels > 1 {
            let maxDim = clampedDivs * (1 << (clampedLevels - 2)) + 2
            let cells = maxDim * maxDim * maxDim
            if cells <= Self.lodCellCap {
                self.lodGridCellCount = cells
            } else {
                print("[GPUPacker] LOD grid \(cells) cells exceeds cap \(Self.lodCellCap); GPU LOD disabled (levels left at finest).")
                self.lodGridCellCount = 0
            }
        } else {
            self.lodGridCellCount = 0
        }

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
        pBucketLOD = try pipeline("packBucketLOD")
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
        let numBatches = cloud.prepareForGPUPack(pointCount: count)
        encodePack(
            positions: positions, colors: colors, count: count,
            firstSlot: 0, numBatches: numBatches, scratch: scratch(0, count: count),
            into: cloud, commandBuffer: commandBuffer
        )
        commandBuffer.addCompletedHandler { _ in
            cloud.adoptGPUBatchBounds(numBatches: numBatches)
        }
    }

    /// Pack one chunk of GPU-resident points into a **contiguous run of pool
    /// slots** starting at `firstSlot`, leaving all other slots untouched —
    /// the streaming building block. The caller is responsible for reserving
    /// `ceil(count / pointsPerBatch)` free contiguous slots and for the CPU
    /// mirror bookkeeping (see ``ComputeRasteriserPointCloud/addRawChunksGPU(packer:queue:chunks:)``).
    ///
    /// Decode-identical to ``pack(positions:colors:count:into:commandBuffer:)``
    /// — only the destination slot range differs (via per-buffer offsets and a
    /// `slotBase` that makes each batch's `firstPoint` absolute).
    ///
    /// - Parameters:
    ///   - positions: `float3` positions for this chunk (16-byte stride, shared).
    ///   - colors: `float4` RGBA `[0,1]` for this chunk (shared).
    ///   - count: number of points in this chunk.
    ///   - firstSlot: pool slot index of this chunk's first batch.
    ///   - cloud: destination pool (must already have the slots reserved).
    ///   - commandBuffer: caller commits; many `packChunk` calls may share one.
    public func packChunk(
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int,
        firstSlot: Int,
        scratchIndex: Int = 0,
        into cloud: ComputeRasteriserPointCloud,
        commandBuffer: MTLCommandBuffer
    ) {
        guard count > 0 else { return }
        let ppb = cloud.capacity.pointsPerBatch
        let numBatches = max(1, (count + ppb - 1) / ppb)
        encodePack(
            positions: positions, colors: colors, count: count,
            firstSlot: firstSlot, numBatches: numBatches,
            scratch: scratch(scratchIndex, count: count),
            into: cloud, commandBuffer: commandBuffer
        )
    }

    /// Encode all pack stages for `count` points into the slot range
    /// `[firstSlot, firstSlot + numBatches)`. Output pool buffers are bound at
    /// the slot's byte offset so the kernels (which index locally, 0..count and
    /// 0..numBatches) land in the right place; `slotBase` makes the per-batch
    /// `firstPoint` absolute. No CPU bookkeeping or completion handler here.
    private func encodePack(
        positions: MTLBuffer,
        colors: MTLBuffer,
        count: Int,
        firstSlot: Int,
        numBatches: Int,
        scratch: PackScratch,
        into cloud: ComputeRasteriserPointCloud,
        commandBuffer: MTLCommandBuffer
    ) {
        let ppb = cloud.capacity.pointsPerBatch
        precondition(ppb <= 65535, "pointsPerBatch must fit the uint16 LOD prefix counts")

        guard let keysA = scratch.keysA, let keysB = scratch.keysB,
              let indicesA = scratch.indicesA, let indicesB = scratch.indicesB,
              let sortedPos = scratch.sortedPos, let boundsBuf = scratch.boundsBuf,
              let histBuf = scratch.histBuf, let tileOffsetBuf = scratch.tileOffsetBuf,
              let digitTotalBuf = scratch.digitTotalBuf, let digitBaseBuf = scratch.digitBaseBuf,
              let xyzLow = cloud.xyzLowBuffer, let xyzMed = cloud.xyzMedBuffer,
              let xyzHigh = cloud.xyzHighBuffer, let colorsOut = cloud.colorsBuffer,
              let levels = cloud.levelsBuffer, let batches = cloud.batchesBuffer
        else { return }

        // Byte offsets into the pool for this chunk's slot run.
        let pointBase = firstSlot * ppb
        let xyzOff = pointBase * MemoryLayout<UInt32>.stride
        let colOff = pointBase * MemoryLayout<UInt32>.stride
        let lvlOff = pointBase * MemoryLayout<UInt8>.stride
        let batchOff = firstSlot * MemoryLayout<RasterBatch>.stride

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
            lodVoxelScale: 1.0,
            slotBase: UInt32(firstSlot)
        )

        // 1. Chunk-local bounds.
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
        // After 4 swaps the sorted indices are back in indicesA (== iIn);
        // keysB/indicesB are dead from here on and double as bucket-stage
        // scratch (Morton-order levels / bucketed permutation).
        let sortedIndices = iIn

        // 4. Gather sorted positions into Morton order. The colorsOut write is
        //    redundant (the bucketed re-gather in stage 8 overwrites it), but
        //    reusing the kernel beats a position-only variant — pack is not
        //    per-frame work.
        encode(commandBuffer, pGather, threads: count) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(colors, offset: 0, index: 1)
            e.setBuffer(sortedIndices, offset: 0, index: 2)
            e.setBuffer(sortedPos, offset: 0, index: 3)
            e.setBuffer(colorsOut, offset: colOff, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }

        // 5. LOD voxel occupancy → Morton-order levels in scratch (keysB, one
        //    uchar per point). Must run on the PRE-bucket order: atomic_min
        //    "lowest sorted index per voxel wins" has to see the same order the
        //    CPU's occupancy walk sees.
        let mortonLevels = keysB
        encode(commandBuffer, pLODInit, threads: count) { e in
            e.setBuffer(mortonLevels, offset: 0, index: 0)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 1)
        }
        if lodLevels > 1, let lodGrid = scratch.lodGrid {
            for level in 0 ..< (lodLevels - 1) {
                params.level = UInt32(level)
                params.lodVoxelScale = powf(0.5, Float(level))
                if let blit = commandBuffer.makeBlitCommandEncoder() {
                    blit.label = "GPUPacker.LODGridFill"
                    blit.fill(buffer: lodGrid, range: 0 ..< (lodGridCellCount * 4), value: 0xff)
                    blit.endEncoding()
                }
                encode(commandBuffer, pLODClaim, threads: count) { e in
                    e.setBuffer(sortedPos, offset: 0, index: 0)
                    e.setBuffer(mortonLevels, offset: 0, index: 1)
                    e.setBuffer(lodGrid, offset: 0, index: 2)
                    e.setBuffer(boundsBuf, offset: 0, index: 3)
                    e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
                }
                encode(commandBuffer, pLODAssign, threads: count) { e in
                    e.setBuffer(sortedPos, offset: 0, index: 0)
                    e.setBuffer(mortonLevels, offset: 0, index: 1)
                    e.setBuffer(lodGrid, offset: 0, index: 2)
                    e.setBuffer(boundsBuf, offset: 0, index: 3)
                    e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 4)
                }
            }
        }

        // 6. Per-batch AABB → batch metadata at batches[firstSlot..]. The AABB
        //    is permutation-invariant within a batch slice, so it can run on
        //    the Morton-ordered positions; the bucket stage patches p3..p6.
        encodeGroups(commandBuffer, pBatchAABB, groups: numBatches, threadsPerGroup: 256) { e in
            e.setBuffer(sortedPos, offset: 0, index: 0)
            e.setBuffer(batches, offset: batchOff, index: 1)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 2)
        }

        // 7. Stable per-batch level bucketing → permuted indices (indicesB),
        //    final-order levels at the slot run, cumulative counts into p3..p6.
        encode(commandBuffer, pBucketLOD, threads: numBatches) { e in
            e.setBuffer(mortonLevels, offset: 0, index: 0)
            e.setBuffer(sortedIndices, offset: 0, index: 1)
            e.setBuffer(indicesB, offset: 0, index: 2)
            e.setBuffer(levels, offset: lvlOff, index: 3)
            e.setBuffer(batches, offset: batchOff, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }

        // 8. Re-gather with the bucketed permutation → final point order in
        //    sortedPos + packed colors at the slot run.
        encode(commandBuffer, pGather, threads: count) { e in
            e.setBuffer(positions, offset: 0, index: 0)
            e.setBuffer(colors, offset: 0, index: 1)
            e.setBuffer(indicesB, offset: 0, index: 2)
            e.setBuffer(sortedPos, offset: 0, index: 3)
            e.setBuffer(colorsOut, offset: colOff, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }

        // 9. Quantize → xyzLow/Med/High at the slot run.
        encode(commandBuffer, pQuantize, threads: count) { e in
            e.setBuffer(sortedPos, offset: 0, index: 0)
            e.setBuffer(batches, offset: batchOff, index: 1)
            e.setBuffer(xyzLow, offset: xyzOff, index: 2)
            e.setBuffer(xyzMed, offset: xyzOff, index: 3)
            e.setBuffer(xyzHigh, offset: xyzOff, index: 4)
            e.setBytes(&params, length: MemoryLayout<PackParams>.stride, index: 5)
        }
    }

    // MARK: - Scratch

    /// Return scratch set `index` (creating sets as needed), grown to fit
    /// `count` points. Sets are independent so chunks assigned to different
    /// sets in one command buffer overlap on the GPU instead of serialising.
    private func scratch(_ index: Int, count: Int) -> PackScratch {
        while scratchSets.count <= index {
            scratchSets.append(PackScratch())
        }
        let s = scratchSets[index]
        s.ensure(device: context.device, count: count,
                 radixTileSize: Self.radixTileSize, lodGridCellCount: lodGridCellCount)
        return s
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

/// One independent set of GPU-private scratch buffers for a single in-flight
/// pack: radix keys/indices/sorted-positions, histogram/scan temporaries, and
/// the dense LOD voxel grid. ``GPUPacker`` keeps a ring of these so several
/// chunks packed into one command buffer use *different* scratch and overlap on
/// the GPU rather than serialising under Metal hazard tracking.
///
/// After the radix sort's final pass, `keysB`/`indicesB` are dead and are
/// reused by the LOD-bucket stage: `keysB` holds the Morton-order LOD levels
/// (one `uchar` per point, well within its 4 bytes/point) and `indicesB`
/// receives the bucketed index permutation for the second gather.
private final class PackScratch {
    var capacityPoints = 0
    var keysA, keysB, indicesA, indicesB: MTLBuffer?
    var sortedPos, boundsBuf: MTLBuffer?
    var histBuf, tileOffsetBuf, digitTotalBuf, digitBaseBuf: MTLBuffer?
    var lodGrid: MTLBuffer?

    /// Grow the per-point buffers to `count` (no-op if already big enough) and
    /// lazily allocate the fixed-size and LOD-grid buffers.
    func ensure(device: MTLDevice, count: Int, radixTileSize: Int, lodGridCellCount: Int) {
        if count > capacityPoints {
            let uintLen = count * MemoryLayout<UInt32>.stride
            let numTiles = (count + radixTileSize - 1) / radixTileSize
            keysA = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            keysB = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            indicesA = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            indicesB = device.makeBuffer(length: uintLen, options: .storageModePrivate)
            sortedPos = device.makeBuffer(length: count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
            histBuf = device.makeBuffer(length: numTiles * 256 * 4, options: .storageModePrivate)
            tileOffsetBuf = device.makeBuffer(length: numTiles * 256 * 4, options: .storageModePrivate)
            capacityPoints = count
        }
        if boundsBuf == nil {
            boundsBuf = device.makeBuffer(length: 6 * 4, options: .storageModePrivate)
            digitTotalBuf = device.makeBuffer(length: 256 * 4, options: .storageModePrivate)
            digitBaseBuf = device.makeBuffer(length: 256 * 4, options: .storageModePrivate)
        }
        if lodGridCellCount > 0, lodGrid == nil {
            lodGrid = device.makeBuffer(length: lodGridCellCount * 4, options: .storageModePrivate)
        }
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

    /// Pack several GPU-resident raw-point chunks into free pool slots in one
    /// command buffer, **adding** them to the resident set (existing slots are
    /// left untouched). Each chunk takes a contiguous slot run; chunks for which
    /// no run is free are skipped (and reported via the short return count).
    /// Synchronous: commits and waits, then syncs the CPU mirror.
    ///
    /// This is the per-frame streaming building block proven in Phase 1 — the
    /// CPU `ChunkPacker` + ``addBatches`` replacement. Returns the first-slot of
    /// each chunk that was placed, in input order (shorter than `chunks` if the
    /// pool ran out of contiguous runs).
    ///
    /// - Parameters:
    ///   - packer: a configured ``GPUPacker``.
    ///   - queue: command queue to submit on.
    ///   - chunks: `(positions, colors, count)` per chunk; buffers are
    ///     `float3`/`float4` `.storageModeShared`, caller-owned.
    /// - Parameter maxConcurrent: number of independent scratch sets to ring
    ///   over so chunks in one command buffer can overlap on the GPU. Measured
    ///   benefit is marginal (~6% at the sweet spot, regresses past 4) because a
    ///   single ~140k-point pack already saturates the GPU's sort/LOD-grid
    ///   bandwidth — and each extra set costs a ~tens-of-MB LOD grid. Defaults
    ///   to `1` (shared scratch, no extra memory); raise only if profiling a
    ///   specific workload shows overlap helps.
    @discardableResult
    public func addRawChunksGPU(
        packer: GPUPacker,
        queue: MTLCommandQueue,
        chunks: [(positions: MTLBuffer, colors: MTLBuffer, count: Int)],
        maxConcurrent: Int = 1
    ) -> [Int] {
        let ppb = capacity.pointsPerBatch
        var placed: [(firstSlot: Int, numBatches: Int, count: Int, index: Int)] = []
        for (i, c) in chunks.enumerated() where c.count > 0 {
            let nb = max(1, (c.count + ppb - 1) / ppb)
            guard let firstSlot = reserveContiguousSlots(nb) else { continue }
            placed.append((firstSlot, nb, c.count, i))
        }
        guard !placed.isEmpty, let commandBuffer = queue.makeCommandBuffer() else { return [] }
        commandBuffer.label = "GPUPacker.addRawChunksGPU"
        // Ring over `maxConcurrent` scratch sets so chunks land in different
        // scratch and can overlap on the GPU (instead of serialising on shared
        // sort/LOD buffers). Each set costs ~one max-chunk's sort scratch + one
        // dense LOD grid, so the ring is bounded.
        let sets = max(1, min(maxConcurrent, placed.count))
        for (n, p) in placed.enumerated() {
            packer.packChunk(
                positions: chunks[p.index].positions, colors: chunks[p.index].colors,
                count: p.count, firstSlot: p.firstSlot, scratchIndex: n % sets,
                into: self, commandBuffer: commandBuffer
            )
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        for p in placed { finalizeGPUChunk(firstSlot: p.firstSlot, numBatches: p.numBatches, count: p.count) }
        commitBatchUpdates()
        return placed.map(\.firstSlot)
    }
}
#endif
