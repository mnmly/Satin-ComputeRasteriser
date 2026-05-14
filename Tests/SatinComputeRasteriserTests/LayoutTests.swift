import Foundation
import Testing
@testable import SatinComputeRasteriser

@Test func sharedTypeStridesMatchMetalExpectations() {
    #expect(ComputeRasteriserLayout.rasterBatchStride == 64)
    #expect(ComputeRasteriserLayout.rasterFileStride == 256)
    #expect(ComputeRasteriserLayout.rasterPixelStride == 48)
    #expect(ComputeRasteriserLayout.visibleBatchStride == 16)
    #expect(MemoryLayout<UInt64>.stride == 8)
}

// Guards the cross-package memcpy from SwiftPDAL's StreamingRasterBatch into
// this package's RasterBatch. Field offsets here must match the streaming
// source's struct exactly; mismatches would silently corrupt batch metadata.
@Test func rasterBatchFieldOffsetsMatchStreamingLayout() {
    #expect(MemoryLayout<RasterBatch>.offset(of: \.state) == 0)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.minX) == 4)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.maxZ) == 24)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.numPoints) == 28)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.firstPoint) == 32)
    #expect(MemoryLayout<RasterBatch>.offset(of: \.fileIndex) == 36)
}

@Test func rasterBatchDefaultsToResident() {
    let batch = RasterBatch(min: .zero, max: .one, numPoints: 1, firstPoint: 0)
    #expect(batch.state == 1)
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
    let rgb = Set(packed.colors.map { $0 & 0x00ffffff })
    #expect(rgb == [0x000000ff, 0x0000ff00, 0x00ff0000])
    #expect(packed.levels.count == 3)
}

@Test func packAssignsLODLevelsInSeparateBufferAndCoversAllPoints() {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    for z in 0 ..< 8 {
        for y in 0 ..< 8 {
            for x in 0 ..< 8 {
                positions.append(SIMD3<Float>(Float(x), Float(y), Float(z)))
                colors.append(SIMD4<Float>(1, 1, 1, 1))
            }
        }
    }

    let packed = PackedPointCloudFixtures.pack(
        positions: positions,
        colors: colors,
        pointsPerBatch: 128,
        lodLevels: 4,
        coarseVoxelDivisions: 2
    )

    #expect(packed.levels.count == positions.count)
    var levelCounts = [Int](repeating: 0, count: 8)
    for level in packed.levels {
        levelCounts[Int(level)] += 1
    }
    let assigned = levelCounts[0] + levelCounts[1] + levelCounts[2] + levelCounts[3]
    #expect(assigned == positions.count)
    #expect(levelCounts[0] > 0)
    #expect(levelCounts[0] < positions.count)

    for c in packed.colors {
        #expect(c >> 24 == 0xff)
    }
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
        0xff0000ff: 1,
        0xff00ff00: 1,
        0xffff0000: 1,
        0xff00ffff: 1,
        0xffffffff: 1,
    ]
    var actual: [UInt32: Int] = [:]
    for c in packed.colors { actual[c, default: 0] += 1 }
    #expect(actual == expected)
}
