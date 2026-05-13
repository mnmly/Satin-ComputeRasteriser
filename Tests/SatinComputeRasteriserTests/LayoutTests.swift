import Foundation
import Testing
@testable import SatinComputeRasteriser

@Test func sharedTypeStridesMatchMetalExpectations() {
    #expect(ComputeRasteriserLayout.rasterBatchStride == 64)
    #expect(ComputeRasteriserLayout.rasterFileStride == 256)
    #expect(ComputeRasteriserLayout.rasterPixelStride == 48)
    #expect(MemoryLayout<UInt64>.stride == 8)
}

@Test func fixturePackingProducesConsistentCounts() {
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 4)
    #expect(packed.pointCount == 64)
    #expect(packed.colors.count == packed.pointCount)
    #expect(packed.xyzLow.count == packed.pointCount)
    #expect(packed.xyzMed.count == packed.pointCount)
    #expect(packed.xyzHigh.count == packed.pointCount)
    #expect(!packed.batches.isEmpty)
    #expect(packed.files.count == 1)
}

@Test func asciiPLYLoaderPacksVertexPositionsAndColors() throws {
    let ply = """
    ply
    format ascii 1.0
    element vertex 3
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    0 0 0 255 0 0
    1 0 0 0 255 0
    0 1 0 0 0 255
    """

    let packed = try PLYPointCloudLoader.parse(Data(ply.utf8), pointsPerBatch: 2)
    #expect(packed.pointCount == 3)
    #expect(packed.batchCount == 2)
    let colorSet = Set(packed.colors)
    #expect(colorSet == [0x000000ff, 0x0000ff00, 0x00ff0000])
}

@Test func packPreservesPointCountAndColorMultisetUnderMortonOrdering() {
    let positions: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, 0),
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(0, 0, 1),
        SIMD3<Float>(1, 1, 1),
    ]
    let colors: [SIMD4<Float>] = [
        SIMD4<Float>(1, 0, 0, 1),
        SIMD4<Float>(0, 1, 0, 1),
        SIMD4<Float>(0, 0, 1, 1),
        SIMD4<Float>(1, 1, 0, 1),
        SIMD4<Float>(1, 1, 1, 1),
    ]

    let packed = PackedPointCloudFixtures.pack(positions: positions, colors: colors, pointsPerBatch: 2)
    #expect(packed.pointCount == 5)
    #expect(packed.batchCount == 3)

    let expected: [UInt32: Int] = [
        0x000000ff: 1,
        0x0000ff00: 1,
        0x00ff0000: 1,
        0x0000ffff: 1,
        0x00ffffff: 1,
    ]
    var actual: [UInt32: Int] = [:]
    for c in packed.colors { actual[c, default: 0] += 1 }
    #expect(actual == expected)
}
