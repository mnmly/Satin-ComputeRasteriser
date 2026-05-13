import Foundation
import Testing
@testable import SatinComputeRasteriser

@Test func nearestPointReferenceUsesReverseZDepth() {
    let packed = PackedPointCloudFixtures.pack(
        positions: [
            SIMD3<Float>(0.0, 0.0, 0.25),
            SIMD3<Float>(0.0, 0.0, 0.75),
        ],
        colors: [
            SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(0.0, 1.0, 0.0, 1.0),
        ],
        pointsPerBatch: 2
    )

    let result = NearestPointCPUReference.render(packed: packed, width: 4, height: 4)
    let centerPixel = 1 * 4 + 2

    #expect(result.indices[centerPixel] == 1)
    #expect(result.colorBuffer(from: packed.colors)[centerPixel] == 0xff00ff00)
}

@Test func nearestPointReferenceKeepsLowestPointIndexOnDepthTie() {
    let packed = PackedPointCloudFixtures.pack(
        positions: [
            SIMD3<Float>(0.0, 0.0, 0.5),
            SIMD3<Float>(0.0, 0.0, 0.5),
        ],
        colors: [
            SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(0.0, 1.0, 0.0, 1.0),
        ],
        pointsPerBatch: 2
    )

    let result = NearestPointCPUReference.render(packed: packed, width: 4, height: 4)
    let centerPixel = 1 * 4 + 2

    #expect(result.indices[centerPixel] == 0)
    #expect(result.colorBuffer(from: packed.colors)[centerPixel] == 0xff0000ff)
}

@Test func nearestPointReferenceRejectsBackgroundAndOutOfClipPoints() {
    let packed = PackedPointCloudFixtures.pack(
        positions: [
            SIMD3<Float>(2.0, 0.0, 0.5),
            SIMD3<Float>(0.0, 0.0, -0.5),
        ],
        colors: [
            SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(0.0, 1.0, 0.0, 1.0),
        ],
        pointsPerBatch: 2
    )

    let result = NearestPointCPUReference.render(packed: packed, width: 4, height: 4)

    #expect(result.indices.allSatisfy { $0 == UInt32.max })
    #expect(result.depths.allSatisfy { $0 == 0 })
}
