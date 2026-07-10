import Foundation
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
/// delta, free evicted slots, upload up to ``maxChunkUploadsPerTick`` new
/// chunks (the remainder carries over to later ticks), and commit once at
/// the end of the tick so a residency churn costs one buffer flush rather
/// than one per chunk.
public final class StreamingAdapter {
    private let source: any StreamingPointCloudSource
    private let cloud: ComputeRasteriserPointCloud
    private var slotsByChunk: [ChunkID: [Int]] = [:]
    /// Newly-streamed chunks awaiting upload, drained ``maxChunkUploadsPerTick``
    /// at a time per ``update(camera:viewport:)`` call. Consumed entries stay
    /// in place behind ``pendingCursor`` and are compacted away lazily, so a
    /// burst drain doesn't pay an O(n) `removeFirst` per chunk.
    private var pendingAdds: [ResidentChunk] = []
    private var pendingCursor: Int = 0

    /// Number of chunks currently resident in the slot pool.
    public private(set) var residentChunks: Int = 0
    /// Sum of points across resident chunks (mirrors `cloud.pointCount`).
    public private(set) var residentPoints: Int = 0
    /// Chunks that have been drained from the source but not yet uploaded to
    /// the slot pool — the throttle carry-over plus any chunk parked by
    /// back-pressure when the pool was full. The source's residency model
    /// counts these as on-GPU, so a healthy adapter satisfies
    /// `sourceBelievedResident == residentChunks + pendingUploadCount`.
    internal var pendingUploadCount: Int { pendingAdds.count - pendingCursor }
    /// Last non-fatal error surfaced during ``update(camera:viewport:)`` —
    /// typically a "slot pool full" notice when the resident budget is
    /// undersized for what the scorer wants to bring in.
    public private(set) var lastError: String?
    /// Bounds how many newly-streamed chunks are uploaded (memcpy'd to the
    /// GPU slot pool) per ``update(camera:viewport:)`` call. A residency
    /// burst of dozens of chunks would otherwise upload — and hitch — in a
    /// single frame; chunks past the cap carry over to subsequent ticks.
    /// Set to `Int.max` to restore the old unbounded, upload-everything-now
    /// behavior.
    public var maxChunkUploadsPerTick: Int = 4

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

        let delta = source.pollLatest()
        guard delta != nil || pendingCursor < pendingAdds.count else { return }

        // Evictions first so freed slots are available for the new chunks.
        // Defer the GPU-side flush — one commitBatchUpdates() at the end
        // of the tick covers both the removes and all the adds, instead of
        // re-uploading the full batch mirror per chunk.
        var dirty = false
        if let delta {
            var slotsToFree: [Int] = []
            for id in delta.removed {
                if let slots = slotsByChunk.removeValue(forKey: id) {
                    slotsToFree.append(contentsOf: slots)
                }
            }
            if !delta.removed.isEmpty {
                // The GPU never saw a still-pending add for a chunk that got
                // evicted before its turn — drop it rather than resurrect it.
                // Compact first so the filter only sees unconsumed chunks.
                if pendingCursor > 0 {
                    pendingAdds.removeFirst(pendingCursor)
                    pendingCursor = 0
                }
                let removedIDs = Set(delta.removed)
                pendingAdds.removeAll { removedIDs.contains($0.id) }
            }
            if !slotsToFree.isEmpty {
                cloud.removeBatches(slots: slotsToFree, commit: false)
                dirty = true
            }
            pendingAdds.append(contentsOf: delta.added)
        }

        // Upload at most `maxChunkUploadsPerTick` chunks this frame; the rest
        // stay queued and upload on subsequent ticks. Bounds the per-frame
        // main-thread memcpy cost when a residency burst brings in dozens of
        // chunks at once.
        var uploaded = 0
        while uploaded < maxChunkUploadsPerTick, pendingCursor < pendingAdds.count {
            let chunk = pendingAdds[pendingCursor]
            let batches = chunk.batches.map(Self.toRasterBatch)
            if cloud.freeSlotCount < batches.count {
                // A chunk wider than the entire pool can never fit — that's a
                // configuration error (the pool is smaller than a single
                // chunk). Drop it permanently, otherwise it wedges the head of
                // the queue forever and starves every chunk behind it.
                if batches.count > cloud.batchCount {
                    lastError = "chunk needs \(batches.count) slots but pool capacity is \(cloud.batchCount); raise maxResidentBatches"
                    pendingCursor += 1
                    continue
                }
                // The pool is merely full right now. Do NOT consume the chunk:
                // leave it at the cursor so a future tick retries it once an
                // eviction frees a slot, and stop draining this tick — nothing
                // will free a slot until then, so burning the rest of the cap
                // (and re-checking every queued chunk) is pointless. Dropping
                // it here instead would strand the chunk permanently: the
                // source's residency model already counts it on the GPU and
                // won't re-emit it until the scorer evicts and re-admits it,
                // leaving a hole in the cloud while the camera holds still.
                lastError = "slot pool full (capacity \(cloud.batchCount)); raise budget or maxResidentBatches"
                break
            }
            pendingCursor += 1
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
            uploaded += 1
        }
        if pendingCursor == pendingAdds.count {
            pendingAdds.removeAll(keepingCapacity: true)
            pendingCursor = 0
        } else if pendingCursor > 64, pendingCursor * 2 > pendingAdds.count {
            pendingAdds.removeFirst(pendingCursor)
            pendingCursor = 0
        }

        if dirty { cloud.commitBatchUpdates() }

        residentChunks = slotsByChunk.count
        residentPoints = cloud.pointCount
    }

    /// Closes the underlying source. Idempotent on the source side.
    public func close() { pendingAdds.removeAll(); pendingCursor = 0; source.close() }

    private static func toRasterBatch(_ s: StreamingRasterBatch) -> RasterBatch {
        var b = RasterBatch(
            min: SIMD3<Float>(s.minX, s.minY, s.minZ),
            max: SIMD3<Float>(s.maxX, s.maxY, s.maxZ),
            numPoints: s.numPoints,
            firstPoint: s.firstPoint,
            fileIndex: s.fileIndex
        )
        b.state = s.state
        // Carry the packed cumulative LOD level counts (padding3…6) through.
        // SwiftPDAL stores each batch level-ascending and fills these so the
        // cull kernel can bound its draw to a LOD threshold; dropping them
        // leaves `cum[7] == 0`, the "unbucketed" sentinel, which forces every
        // streamed chunk to a full-range draw and makes lodBias a no-op on
        // streamed data. padding7/8 are copied too for layout fidelity.
        b.padding3 = s.padding3
        b.padding4 = s.padding4
        b.padding5 = s.padding5
        b.padding6 = s.padding6
        b.padding7 = s.padding7
        b.padding8 = s.padding8
        return b
    }
}
