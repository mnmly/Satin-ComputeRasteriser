import Foundation
import Metal
import Satin
import SatinComputeRasteriser
import SwiftPDAL
import simd

/// Bridges a SwiftPDAL `StreamingPointCloudSource` to a
/// `ComputeRasteriserPointCloud` slot pool.
///
/// The renderer package stays free of PDAL/lazperf; this adapter is the
/// glue layer where the streaming driver and the GPU residency pool meet.
/// Per frame: submit the camera view, drain the driver's `(added, removed)`
/// delta, free evicted slots, upload new chunks, and commit once at the end
/// of the tick so a residency churn costs one buffer flush rather than one
/// per chunk.
public final class StreamingAdapter {
    private let source: any StreamingPointCloudSource
    private let cloud: ComputeRasteriserPointCloud
    private var slotsByChunk: [ChunkID: [Int]] = [:]

    // GPU-pack path (used only when the source emits raw chunks, i.e.
    // StreamingOptions.emitRawPoints). Lazily built from the cloud's context so
    // the packed path stays zero-cost.
    private lazy var gpuPacker: GPUPacker? = try? GPUPacker(context: cloud.context)
    private lazy var packQueue: MTLCommandQueue? = cloud.context.device.makeCommandQueue()

    /// Number of chunks currently resident in the slot pool.
    public private(set) var residentChunks: Int = 0
    /// Sum of points across resident chunks (mirrors `cloud.pointCount`).
    public private(set) var residentPoints: Int = 0
    /// Last non-fatal error surfaced during ``update(camera:viewport:)`` —
    /// typically a "slot pool full" notice when the resident budget is
    /// undersized for what the scorer wants to bring in.
    public private(set) var lastError: String?

    /// Creates an adapter that owns no resources of its own — it merely
    /// routes data between `source` and `cloud`.
    /// - Parameter source: Streaming driver producing chunked point batches.
    /// - Parameter cloud: Destination renderer holding the GPU slot pool.
    public init(
        source: any StreamingPointCloudSource,
        cloud: ComputeRasteriserPointCloud
    ) {
        self.source = source
        self.cloud = cloud
    }

    /// Forwards a byte budget hint to the underlying source.
    public func setBudget(bytes: Int) { source.setBudget(bytes) }

    /// Per-frame tick. Submits the latest camera view and applies whatever
    /// delta the driver published since the last call. Cheap when nothing
    /// changed (one MTLBlit's worth of work at most).
    /// - Parameter camera: Active scene camera; world position and matrices
    ///   are read to score chunk residency.
    /// - Parameter viewport: Render target size in pixels.
    public func update(camera: Camera, viewport: SIMD2<Float>) {
        // FOV-aware pixelScale derived from the camera's projection matrix
        // — see ``ComputeRasteriserProjection/screenSpacePixelScale(viewportHeight:projectionMatrix:)``.
        // The previous `viewport.y * 0.5` was FOV-blind: a 90° camera and a
        // 10° camera got the same value, so the SwiftPDAL scorer made the
        // same residency decision regardless of zoom.
        let scale = ComputeRasteriserProjection.screenSpacePixelScale(
            viewportHeight: viewport.y,
            projectionMatrix: camera.projectionMatrix
        )
        let view = StreamingCameraView(
            position: camera.worldPosition,
            viewProjection: camera.projectionMatrix * camera.viewMatrix,
            pixelScale: scale
        )
        source.submit(view: view)

        guard let delta = source.pollLatest() else { return }

        // Evictions first so freed slots are available for the new chunks.
        // Defer the GPU-side flush — one commitBatchUpdates() at the end
        // of the tick covers both the removes and all the adds, instead of
        // re-uploading the full batch mirror per chunk.
        var dirty = false
        var slotsToFree: [Int] = []
        for id in delta.removed {
            if let slots = slotsByChunk.removeValue(forKey: id) {
                slotsToFree.append(contentsOf: slots)
            }
        }
        if !slotsToFree.isEmpty {
            cloud.removeBatches(slots: slotsToFree, commit: false)
            dirty = true
        }

        // Raw chunks (StreamingOptions.emitRawPoints) carry un-packed points the
        // GPU packs; everything else takes the CPU-packed addBatches path. A
        // single source emits one or the other, but we branch per chunk so both
        // are supported.
        var rawInputs: [(positions: MTLBuffer, colors: MTLBuffer, count: Int)] = []
        var rawMeta: [(id: ChunkID, count: Int)] = []
        let device = cloud.context.device

        for chunk in delta.added {
            if let rawPositions = chunk.rawPositions, let rawColors = chunk.rawColors {
                let count = chunk.totalPointCount
                guard
                    let posBuf = rawPositions.withUnsafeBytes({ raw in
                        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
                    }),
                    let colBuf = rawColors.withUnsafeBytes({ raw in
                        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
                    })
                else { continue }
                rawInputs.append((posBuf, colBuf, count))
                rawMeta.append((chunk.id, count))
                continue
            }

            let batches = chunk.batches.map(Self.toRasterBatch)
            guard cloud.freeSlotCount >= batches.count else {
                lastError = "slot pool full (capacity \(cloud.batchCount)); raise budget or maxResidentBatches"
                continue
            }
            let slots = cloud.addBatches(
                positionsXYZLow: chunk.xyzLow,
                positionsXYZMed: chunk.xyzMed,
                positionsXYZHigh: chunk.xyzHigh,
                colors: chunk.colors,
                levels: chunk.levels,
                batches: batches,
                commit: false
            )
            slotsByChunk[chunk.id] = slots
            dirty = true
        }

        // Pack this frame's raw chunks in one GPU dispatch.
        if !rawInputs.isEmpty, let packer = gpuPacker, let queue = packQueue {
            let ppb = cloud.capacity.pointsPerBatch
            let firstSlots = cloud.addRawChunksGPU(packer: packer, queue: queue, chunks: rawInputs)
            for (i, meta) in rawMeta.enumerated() {
                let firstSlot = firstSlots[i]
                guard firstSlot >= 0 else {
                    lastError = "slot pool full (capacity \(cloud.batchCount)); raise budget or maxResidentBatches"
                    continue
                }
                let nb = max(1, (meta.count + ppb - 1) / ppb)
                slotsByChunk[meta.id] = Array(firstSlot ..< (firstSlot + nb))
            }
            // addRawChunksGPU already flushed the batch mirror; no extra commit
            // needed for the raw chunks, but a pending evict/packed change still
            // needs the trailing flush below.
        }

        if dirty { cloud.commitBatchUpdates() }

        residentChunks = slotsByChunk.count
        residentPoints = cloud.pointCount
    }

    /// Closes the underlying source. Idempotent on the source side.
    public func close() { source.close() }

    private static func toRasterBatch(_ s: StreamingRasterBatch) -> RasterBatch {
        var b = RasterBatch(
            min: SIMD3<Float>(s.minX, s.minY, s.minZ),
            max: SIMD3<Float>(s.maxX, s.maxY, s.maxZ),
            numPoints: s.numPoints,
            firstPoint: s.firstPoint,
            fileIndex: s.fileIndex
        )
        b.state = s.state
        return b
    }
}
