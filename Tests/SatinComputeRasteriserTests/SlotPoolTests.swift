import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// Exercises the slot-pool incremental-upload path that streaming sources
// (SwiftPDAL's CopcStreamingPointCloudSource) drive. The renderer doesn't
// own the source, so these tests fake the source's output: build a small
// PackedPointCloud with a deterministic shape, then push its batches in
// chunks via addBatches/removeBatches and check that:
//  - slots are assigned LIFO from the free list
//  - residentBatchCount / pointCount track adds and removes
//  - the GPU-visible RasterBatch mirror reflects state and rebased firstPoint
//  - position/color/level slot ranges hold the source bytes byte-for-byte

private func makeContext() throws -> Context {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    return Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
}

@Test func streamingInitAllocatesEmptyPool() throws {
    let context = try makeContext()
    let cap = ComputeRasteriserCapacity(maxResidentBatches: 8, pointsPerBatch: 16)
    let cloud = ComputeRasteriserPointCloud(context: context, capacity: cap)

    #expect(cloud.batchCount == 8)              // cull dispatches over capacity
    #expect(cloud.residentBatchCount == 0)
    #expect(cloud.pointCount == 0)
    #expect(cloud.freeSlotCount == 8)
    #expect(cloud.batchesBuffer != nil)
    #expect(cloud.xyzLowBuffer?.length == 8 * 16 * 4)
    #expect(cloud.colorsBuffer?.length == 8 * 16 * 4)
    #expect(cloud.levelsBuffer?.length == 8 * 16)
}

@Test func addBatchesAssignsSlotsAndRebaseFirstPoint() throws {
    let context = try makeContext()
    let cap = ComputeRasteriserCapacity(maxResidentBatches: 4, pointsPerBatch: 8)
    let cloud = ComputeRasteriserPointCloud(context: context, capacity: cap)

    // Fake "decoded chunk": two batches of 4 points each, contiguous.
    let positions = (0 ..< 8).map { UInt32($0 + 100) }   // sentinel values
    let colors    = (0 ..< 8).map { UInt32($0 + 200) }
    let levels    = (0 ..< 8).map { UInt8($0) }
    let batches = [
        RasterBatch(min: .zero, max: .one, numPoints: 4, firstPoint: 0, fileIndex: 0),
        RasterBatch(min: .zero, max: .one, numPoints: 4, firstPoint: 4, fileIndex: 0),
    ]
    let dataLow = Data(bytes: positions, count: positions.count * 4)
    let dataMed = dataLow
    let dataHigh = dataLow
    let dataColors = Data(bytes: colors, count: colors.count * 4)
    let dataLevels = Data(levels)

    let slots = cloud.addBatches(
        positionsXYZLow: dataLow, positionsXYZMed: dataMed, positionsXYZHigh: dataHigh,
        colors: dataColors, levels: dataLevels, batches: batches
    )

    #expect(slots.count == 2)
    // Free list is LIFO from (0..<4).reversed(), so we pop 0 then 1.
    #expect(slots == [0, 1])
    #expect(cloud.residentBatchCount == 2)
    #expect(cloud.pointCount == 8)
    #expect(cloud.freeSlotCount == 2)

    // GPU-visible RasterBatch mirror: state==1, firstPoint rebased to slot range.
    let mirror = readBatchMirror(buffer: cloud.batchesBuffer!, count: 4)
    #expect(mirror[0].state == 1)
    #expect(mirror[0].firstPoint == 0)        // slot 0 × pointsPerBatch(8) = 0
    #expect(mirror[0].numPoints == 4)
    #expect(mirror[1].state == 1)
    #expect(mirror[1].firstPoint == 8)        // slot 1 × pointsPerBatch(8) = 8
    #expect(mirror[1].numPoints == 4)
    #expect(mirror[2].state == 0)             // untouched slots stay empty
    #expect(mirror[3].state == 0)

    // Position bytes for slot 0 should equal source bytes 0..16; slot 1 = 16..32.
    let xyzLowBytes = readUInt32(buffer: cloud.xyzLowBuffer!, count: 8 * 4)
    #expect(Array(xyzLowBytes[0 ..< 4]) == Array(positions[0 ..< 4]))
    #expect(Array(xyzLowBytes[8 ..< 12]) == Array(positions[4 ..< 8]))

    let levelBytes = readUInt8(buffer: cloud.levelsBuffer!, count: 8 * 8)
    #expect(Array(levelBytes[0 ..< 4]) == Array(levels[0 ..< 4]))
    #expect(Array(levelBytes[8 ..< 12]) == Array(levels[4 ..< 8]))
}

@Test func removeBatchesFreesSlotsAndZeroesState() throws {
    let context = try makeContext()
    let cap = ComputeRasteriserCapacity(maxResidentBatches: 4, pointsPerBatch: 8)
    let cloud = ComputeRasteriserPointCloud(context: context, capacity: cap)

    let positions = [UInt32](repeating: 0, count: 8)
    let dataLow = Data(bytes: positions, count: 32)
    let batches = [
        RasterBatch(min: .zero, max: .one, numPoints: 4, firstPoint: 0, fileIndex: 0),
        RasterBatch(min: .zero, max: .one, numPoints: 4, firstPoint: 4, fileIndex: 0),
    ]
    let slots = cloud.addBatches(
        positionsXYZLow: dataLow, positionsXYZMed: dataLow, positionsXYZHigh: dataLow,
        colors: dataLow, levels: Data(repeating: 0, count: 8), batches: batches
    )

    cloud.removeBatches(slots: [slots[0]])
    #expect(cloud.residentBatchCount == 1)
    #expect(cloud.pointCount == 4)
    #expect(cloud.freeSlotCount == 3)

    let mirror = readBatchMirror(buffer: cloud.batchesBuffer!, count: 4)
    #expect(mirror[slots[0]].state == 0)
    #expect(mirror[slots[0]].numPoints == 0)
    #expect(mirror[slots[1]].state == 1)

    // Newly-freed slot should be at top of LIFO; next addBatches reuses it.
    let reused = cloud.addBatches(
        positionsXYZLow: dataLow, positionsXYZMed: dataLow, positionsXYZHigh: dataLow,
        colors: dataLow, levels: Data(repeating: 0, count: 8),
        batches: [RasterBatch(min: .zero, max: .one, numPoints: 4, firstPoint: 0, fileIndex: 0)]
    )
    #expect(reused == [slots[0]])
}

@Test func legacyPackedInitFitsCapacityToBatchCount() throws {
    let context = try makeContext()
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 8)   // 512 points
    let cloud = ComputeRasteriserPointCloud(context: context, packed: packed)

    #expect(cloud.capacity.maxResidentBatches == packed.batchCount)
    #expect(cloud.residentBatchCount == packed.batchCount)
    #expect(cloud.pointCount == packed.pointCount)
}

// MARK: - GPU buffer readback helpers

private func readBatchMirror(buffer: MTLBuffer, count: Int) -> [RasterBatch] {
    var out = [RasterBatch](repeating: RasterBatch(min: .zero, max: .zero, numPoints: 0, firstPoint: 0, fileIndex: 0, state: 0), count: count)
    let stride = MemoryLayout<RasterBatch>.stride
    let src = buffer.contents().assumingMemoryBound(to: UInt8.self)
    out.withUnsafeMutableBytes { dst in
        for i in 0 ..< count {
            memcpy(dst.baseAddress!.advanced(by: i * stride), src.advanced(by: i * stride), stride)
        }
    }
    return out
}

private func readUInt32(buffer: MTLBuffer, count: Int) -> [UInt32] {
    var out = [UInt32](repeating: 0, count: count)
    let src = buffer.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0 ..< count { out[i] = src[i] }
    return out
}

private func readUInt8(buffer: MTLBuffer, count: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: count)
    let src = buffer.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0 ..< count { out[i] = src[i] }
    return out
}
