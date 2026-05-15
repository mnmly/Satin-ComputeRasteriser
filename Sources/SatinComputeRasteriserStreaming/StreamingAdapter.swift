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
/// delta, free evicted slots, upload new chunks, and commit once at the end
/// of the tick so a residency churn costs one buffer flush rather than one
/// per chunk.
public final class StreamingAdapter {
    private let source: any StreamingPointCloudSource
    private let cloud: ComputeRasteriserPointCloud
    private var slotsByChunk: [ChunkID: [Int]] = [:]

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

        for chunk in delta.added {
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
