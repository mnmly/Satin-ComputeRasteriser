// Glue between SwiftPDAL's CopcStreamingPointCloudSource and the
// renderer's slot pool. Lives in the example app per the streaming
// design (the renderer package never depends on PDAL/LASzip).
//
// Per-frame loop:
//   1. submit camera view to source
//   2. drain pollLatest() — a (added, removed) delta
//   3. removeBatches(slots:) for evictions, addBatches(...) for adds
//   4. record ChunkID → [slot] so future evictions know which slots to free
//
// SwiftPDAL is wrapped in `#if canImport(SwiftPDAL)` so the example app
// keeps building without it. To wire it up:
//   File → Add Package Dependencies… → Add Local… → ../../../../../Personal/SwiftPDAL
//   then add the SwiftPDAL product to this target.

#if canImport(SwiftPDAL)
import Foundation
import Satin
import SatinComputeRasteriser
import SwiftPDAL
import simd

public final class StreamingAdapter {
    private let source: any StreamingPointCloudSource
    private let cloud: ComputeRasteriserPointCloud
    private let pixelScale: Float
    private var slotsByChunk: [ChunkID: [Int]] = [:]

    public private(set) var residentChunks: Int = 0
    public private(set) var residentPoints: Int = 0
    public private(set) var lastError: String?

    public init(
        source: any StreamingPointCloudSource,
        cloud: ComputeRasteriserPointCloud,
        pixelScale: Float = 800
    ) {
        self.source = source
        self.cloud = cloud
        self.pixelScale = pixelScale
    }

    public func setBudget(bytes: Int) { source.setBudget(bytes) }

    /// Per-frame tick. Submits the latest camera view and applies whatever
    /// delta the driver published since the last call. Cheap when nothing
    /// changed (one MTLBlit's worth of work at most).
    public func update(camera: Camera, viewport: SIMD2<Float>) {
        let scale = max(pixelScale, max(viewport.x, viewport.y) * 0.5)
        let view = StreamingCameraView(
            position: camera.worldPosition,
            viewProjection: camera.projectionMatrix * camera.viewMatrix,
            pixelScale: scale
        )
        source.submit(view: view)

        guard let delta = source.pollLatest() else { return }

        // Evictions first so freed slots are available for the new chunks.
        var slotsToFree: [Int] = []
        for id in delta.removed {
            if let slots = slotsByChunk.removeValue(forKey: id) {
                slotsToFree.append(contentsOf: slots)
            }
        }
        if !slotsToFree.isEmpty {
            cloud.removeBatches(slots: slotsToFree)
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
                batches: batches
            )
            slotsByChunk[chunk.id] = slots
        }

        residentChunks = slotsByChunk.count
        residentPoints = cloud.pointCount
    }

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
#endif
