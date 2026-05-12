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
    #expect(packed.colors[0] == 0x000000ff)
    #expect(packed.colors[1] == 0x0000ff00)
    #expect(packed.colors[2] == 0x00ff0000)
}
