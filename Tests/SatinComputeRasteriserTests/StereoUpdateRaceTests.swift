import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// Regression test for the single-shared-`filesBuffer` race that broke stereo
// offline rendering. `updateFiles` is called once per camera each frame —
// in stereo, twice in a row between encodes — and the GPU executes both
// encodes after the second CPU write. If `filesBuffer` has only one slot,
// the right-eye write clobbers the left-eye state and both eyes' compute
// passes see the right-eye viewProjection, producing zero disparity.

@Test func filesBufferSizedForMultipleInFlightUpdates() throws {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 4)
    let cloud = ComputeRasteriserPointCloud(context: context, packed: packed)

    let oneSlotBytes = MemoryLayout<RasterFile>.stride * max(packed.files.count, 1)
    let length = cloud.filesBuffer?.length ?? 0
    #expect(
        length >= oneSlotBytes * 2,
        "filesBuffer must hold ≥2 slots (got \(length) bytes, one slot = \(oneSlotBytes)) — single slot races between sequential updateFiles calls (e.g. stereo left/right)."
    )
}

@Test func updateFilesAdvancesOffsetAndPreservesPreviousSlot() throws {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
    let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 4)
    let cloud = ComputeRasteriserPointCloud(context: context, packed: packed)

    let stride = MemoryLayout<RasterFile>.stride
    let fileCount = packed.files.count
    let slotBytes = stride * fileCount

    // Two distinct viewProjections — values are unrealistic on purpose to make
    // the slot contents trivially recognisable.
    let vpLeft = matrix_identity_float4x4
    var vpRight = matrix_identity_float4x4
    vpRight.columns.3.x = 999

    cloud.updateFiles(viewProjection: vpLeft, modelMatrix: matrix_identity_float4x4)
    let leftOffset = cloud.filesBufferOffset
    let leftBytes = snapshot(buffer: cloud.filesBuffer!, offset: leftOffset, length: slotBytes)

    cloud.updateFiles(viewProjection: vpRight, modelMatrix: matrix_identity_float4x4)
    let rightOffset = cloud.filesBufferOffset

    // 1) The second update must land in a different slot than the first —
    //    otherwise the GPU's left-eye compute reads right-eye data.
    #expect(leftOffset != rightOffset, "filesBufferOffset must advance per update (got both = \(leftOffset))")

    // 2) The LEFT slot's contents must still be the LEFT data after the
    //    RIGHT update — proves the slot the LEFT encode bound is intact.
    let leftAfterRightWrite = snapshot(buffer: cloud.filesBuffer!, offset: leftOffset, length: slotBytes)
    #expect(leftAfterRightWrite == leftBytes, "LEFT slot was overwritten by subsequent RIGHT updateFiles")
}

private func snapshot(buffer: MTLBuffer, offset: Int, length: Int) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: length)
    let src = buffer.contents().advanced(by: offset).assumingMemoryBound(to: UInt8.self)
    for i in 0 ..< length { bytes[i] = src[i] }
    return bytes
}
