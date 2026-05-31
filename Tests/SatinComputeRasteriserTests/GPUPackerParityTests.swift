#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// GPU pack() must be decode-identical to the CPU PackedPointCloudFixtures.pack:
// same Morton order, same per-batch quantization, same LOD levels. These tests
// pack the same input both ways and compare the *decoded* world positions
// (within one quantization step) and LOD levels (exactly), reproducing the
// rasteriser's scr_decodePointAt inverse on the CPU.

private func makeContext() throws -> Context {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    return Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
}

private func unpack10(_ encoded: UInt32, _ shift: UInt32) -> UInt32 { (encoded >> shift) & 1023 }

/// Mirror of `scr_decodePointAt` (DisplacementPass preamble).
private func decode(
    low: UInt32, med: UInt32, high: UInt32, level: UInt8,
    batchMin: SIMD3<Float>, batchMax: SIMD3<Float>
) -> SIMD3<Float> {
    let size = simd_max(batchMax - batchMin, SIMD3<Float>(repeating: 1e-6))
    let l = Int(level) & 0x7
    func axis(_ shift: UInt32) -> (UInt32, UInt32, UInt32) {
        (unpack10(low, shift), unpack10(med, shift), unpack10(high, shift))
    }
    let (lx, mx, hx) = axis(0)
    let (ly, my, hy) = axis(10)
    let (lz, mz, hz) = axis(20)
    let steps: Float
    let q: SIMD3<UInt32>
    if l == 0 {
        q = SIMD3(
            (lx << 20) | (mx << 10) | hx,
            (ly << 20) | (my << 10) | hy,
            (lz << 20) | (mz << 10) | hz
        )
        steps = 1_073_741_824.0
    } else if l == 1 {
        q = SIMD3((lx << 10) | mx, (ly << 10) | my, (lz << 10) | mz)
        steps = 1_048_576.0
    } else {
        q = SIMD3(lx, ly, lz)
        steps = 1024.0
    }
    return SIMD3<Float>(Float(q.x), Float(q.y), Float(q.z)) * (size / steps) + batchMin
}

/// A regular lattice has distinct Morton keys (one point per coarse cell), so
/// the GPU radix sort and CPU comparison sort produce the identical order —
/// letting us compare per index, not just as a multiset.
private func lattice(_ n: Int) -> ([SIMD3<Float>], [SIMD4<Float>]) {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    for z in 0 ..< n {
        for y in 0 ..< n {
            for x in 0 ..< n {
                let fx = Float(x) / Float(n - 1)
                let fy = Float(y) / Float(n - 1)
                let fz = Float(z) / Float(n - 1)
                positions.append(SIMD3<Float>(fx - 0.5, fy - 0.5, fz - 0.5))
                colors.append(SIMD4<Float>(fx, fy, fz, 1.0))
            }
        }
    }
    return (positions, colors)
}

private func sharedBuffer<T>(_ device: MTLDevice, _ array: [T]) -> MTLBuffer {
    array.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}

@Test func gpuPackDecodeMatchesCPUSingleBatch() throws {
    let context = try makeContext()
    let device = context.device
    let (positions, colors) = lattice(16) // 4096 points → single batch
    let count = positions.count

    let cpu = PackedPointCloudFixtures.pack(positions: positions, colors: colors)

    let packer = try GPUPacker(context: context)
    let cloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(maxResidentBatches: 1, pointsPerBatch: computeRasteriserThreadsPerGroup * 80)
    )
    let queue = try #require(device.makeCommandQueue())
    let posBuf = sharedBuffer(device, positions)
    let colBuf = sharedBuffer(device, colors)
    cloud.replacePackedPointCloud(packer: packer, queue: queue, positions: posBuf, colors: colBuf, count: count)

    #expect(cloud.pointCount == count)
    #expect(cloud.residentBatchCount == cpu.batchCount)

    // Read back GPU pool buffers.
    let gLow = cloud.xyzLowBuffer!.contents().bindMemory(to: UInt32.self, capacity: count)
    let gMed = cloud.xyzMedBuffer!.contents().bindMemory(to: UInt32.self, capacity: count)
    let gHigh = cloud.xyzHighBuffer!.contents().bindMemory(to: UInt32.self, capacity: count)
    let gLevels = cloud.levelsBuffer!.contents().bindMemory(to: UInt8.self, capacity: count)
    let gBatches = cloud.batchesBuffer!.contents().bindMemory(to: RasterBatch.self, capacity: cloud.batchCount)

    let cpuBatch = cpu.batches[0]
    let cpuMin = SIMD3<Float>(cpuBatch.minX, cpuBatch.minY, cpuBatch.minZ)
    let cpuMax = SIMD3<Float>(cpuBatch.maxX, cpuBatch.maxY, cpuBatch.maxZ)
    let gBatch = gBatches[0]
    let gMin = SIMD3<Float>(gBatch.minX, gBatch.minY, gBatch.minZ)
    let gMax = SIMD3<Float>(gBatch.maxX, gBatch.maxY, gBatch.maxZ)

    // Per-batch AABB should agree closely.
    #expect(simd_distance(cpuMin, gMin) < 1e-5)
    #expect(simd_distance(cpuMax, gMax) < 1e-5)

    // One quantization step in this batch.
    let step = simd_reduce_max(cpuMax - cpuMin) / 1_073_741_824.0
    let tol = step * 4 // a few steps of slack for float-rounding differences

    var levelMismatches = 0
    var maxPosErr: Float = 0
    for i in 0 ..< count {
        let cpuPos = decode(low: cpu.xyzLow[i], med: cpu.xyzMed[i], high: cpu.xyzHigh[i],
                            level: cpu.levels[i], batchMin: cpuMin, batchMax: cpuMax)
        let gpuPos = decode(low: gLow[i], med: gMed[i], high: gHigh[i],
                            level: gLevels[i], batchMin: gMin, batchMax: gMax)
        maxPosErr = max(maxPosErr, simd_distance(cpuPos, gpuPos))
        if cpu.levels[i] != gLevels[i] { levelMismatches += 1 }
    }

    #expect(maxPosErr <= tol, "max decoded position error \(maxPosErr) exceeded tol \(tol)")
    // Distinct keys → identical sort order → levels must match exactly.
    #expect(levelMismatches == 0, "\(levelMismatches)/\(count) LOD levels differed")
}

@Test func gpuPackMultiBatchLevelDistributionMatches() throws {
    let context = try makeContext()
    let device = context.device
    let (positions, colors) = lattice(40) // 64000 points → multiple batches
    let count = positions.count

    let cpu = PackedPointCloudFixtures.pack(positions: positions, colors: colors)

    let packer = try GPUPacker(context: context)
    let cloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(maxResidentBatches: 1, pointsPerBatch: computeRasteriserThreadsPerGroup * 80)
    )
    let queue = try #require(device.makeCommandQueue())
    cloud.replacePackedPointCloud(
        packer: packer, queue: queue,
        positions: sharedBuffer(device, positions), colors: sharedBuffer(device, colors), count: count
    )

    #expect(cloud.pointCount == count)
    #expect(cloud.residentBatchCount == cpu.batchCount)

    // Per-level counts should match (per-point identity can differ only for
    // equal-key ties straddling a batch boundary, which is render-invisible).
    let gLevels = cloud.levelsBuffer!.contents().bindMemory(to: UInt8.self, capacity: count)
    var cpuHist = [Int](repeating: 0, count: 8)
    var gpuHist = [Int](repeating: 0, count: 8)
    for i in 0 ..< count {
        cpuHist[Int(cpu.levels[i])] += 1
        gpuHist[Int(gLevels[i])] += 1
    }
    for level in 0 ..< 8 {
        let diff = abs(cpuHist[level] - gpuHist[level])
        #expect(diff <= count / 1000, "level \(level): cpu=\(cpuHist[level]) gpu=\(gpuHist[level])")
    }
}
#endif
