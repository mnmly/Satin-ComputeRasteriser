#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// GPU pack() must be decode-identical to the CPU PackedPointCloudFixtures.pack:
// same Morton order, same per-batch quantization, same LOD levels, same stable
// level-ascending bucketing with cumulative counts in padding3...6. These tests
// pack the same input both ways and compare the *decoded* world positions
// (within one quantization step) and LOD levels (exactly, per index),
// reproducing the rasteriser's scr_decodePointAt inverse on the CPU.

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

/// LOD-bucketing invariants on a GPU-packed slot run: per-batch levels stored
/// non-decreasing, the cumulative counts in padding3...6 match a recount of the
/// levels buffer, and padding6 != 0 (cum[7] == numPoints — not the legacy
/// unbucketed sentinel).
private func verifyBucketedBatches(
    _ cloud: ComputeRasteriserPointCloud,
    firstSlot: Int, numBatches: Int, ppb: Int, label: String
) {
    let gBatches = cloud.batchesBuffer!.contents().bindMemory(to: RasterBatch.self, capacity: cloud.batchCount)
    let gLevels = cloud.levelsBuffer!.contents().bindMemory(to: UInt8.self, capacity: cloud.batchCount * ppb)
    for b in 0 ..< numBatches {
        let batch = gBatches[firstSlot + b]
        let base = (firstSlot + b) * ppb
        let n = Int(batch.numPoints)
        var cumulative = [Int](repeating: 0, count: 8)
        var ascending = true
        for i in 0 ..< n {
            cumulative[Int(gLevels[base + i]) & 7] += 1
            if i > 0, gLevels[base + i] < gLevels[base + i - 1] { ascending = false }
        }
        for level in 1 ..< 8 {
            cumulative[level] += cumulative[level - 1]
        }
        #expect(ascending, "\(label) batch \(b): levels not stored level-ascending")
        #expect(batch.lodCumulativeCounts.map(Int.init) == cumulative,
                "\(label) batch \(b): prefix counts \(batch.lodCumulativeCounts) != recount \(cumulative)")
        #expect(batch.padding6 != 0, "\(label) batch \(b): padding6 == 0 (unbucketed sentinel)")
    }
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
    // Distinct keys → identical sort order → identical LOD assignment →
    // identical stable bucketing → levels must match exactly per index.
    #expect(levelMismatches == 0, "\(levelMismatches)/\(count) LOD levels differed")

    // GPU prefix counts must match the CPU batch's exactly, and satisfy the
    // bucketing invariants against the GPU's own levels.
    #expect(gBatch.lodCumulativeCounts == cpuBatch.lodCumulativeCounts)
    verifyBucketedBatches(cloud, firstSlot: 0, numBatches: cpu.batchCount,
                          ppb: cloud.capacity.pointsPerBatch, label: "single")
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

    verifyBucketedBatches(cloud, firstSlot: 0, numBatches: cpu.batchCount,
                          ppb: cloud.capacity.pointsPerBatch, label: "multi")
}

// MARK: - Slot-targeted (streaming) pack parity
//
// addRawChunksGPU packs each chunk into a contiguous run of pool slots without
// disturbing existing residents — the Phase-1 streaming building block. For a
// lattice (distinct Morton keys) the sort order is deterministic, so each GPU
// batch contains exactly the CPU batch's points; we assert per-batch AABB
// equality (tie-independent), absolute firstPoint, numPoints, the LOD level
// histogram, and the bucketing invariants — read at the chunk's slot offset in
// the pool.

private func verifyChunkAtSlot(
    _ cloud: ComputeRasteriserPointCloud, _ cpu: PackedPointCloud,
    firstSlot: Int, ppb: Int, label: String
) {
    let count = cpu.xyzLow.count
    let nb = cpu.batchCount
    let poolBatches = cloud.batchCount
    let gBatches = cloud.batchesBuffer!.contents().bindMemory(to: RasterBatch.self, capacity: poolBatches)
    let gLevels = cloud.levelsBuffer!.contents().bindMemory(to: UInt8.self, capacity: poolBatches * ppb)
    let base = firstSlot * ppb

    for b in 0 ..< nb {
        let c = cpu.batches[b]
        let g = gBatches[firstSlot + b]
        let cMin = SIMD3<Float>(c.minX, c.minY, c.minZ), cMax = SIMD3<Float>(c.maxX, c.maxY, c.maxZ)
        let gMin = SIMD3<Float>(g.minX, g.minY, g.minZ), gMax = SIMD3<Float>(g.maxX, g.maxY, g.maxZ)
        #expect(simd_distance(cMin, gMin) < 1e-5, "\(label) batch \(b) min: cpu \(cMin) gpu \(gMin)")
        #expect(simd_distance(cMax, gMax) < 1e-5, "\(label) batch \(b) max: cpu \(cMax) gpu \(gMax)")
        #expect(g.firstPoint == UInt32((firstSlot + b) * ppb), "\(label) batch \(b) firstPoint not absolute (\(g.firstPoint))")
        #expect(g.numPoints == c.numPoints, "\(label) batch \(b) numPoints cpu=\(c.numPoints) gpu=\(g.numPoints)")
        #expect(g.state == 1)
    }

    var cpuHist = [Int](repeating: 0, count: 8)
    var gpuHist = [Int](repeating: 0, count: 8)
    for i in 0 ..< count {
        cpuHist[Int(cpu.levels[i]) & 7] += 1
        gpuHist[Int(gLevels[base + i]) & 7] += 1
    }
    for level in 0 ..< 8 {
        #expect(abs(cpuHist[level] - gpuHist[level]) <= count / 1000,
                "\(label) level \(level): cpu=\(cpuHist[level]) gpu=\(gpuHist[level])")
    }

    verifyBucketedBatches(cloud, firstSlot: firstSlot, numBatches: nb, ppb: ppb, label: label)
}

@Test func gpuPackChunkSlotTargetedNonDestructive() throws {
    let context = try makeContext()
    let device = context.device
    let queue = try #require(device.makeCommandQueue())
    let packer = try GPUPacker(context: context)
    let ppb = computeRasteriserThreadsPerGroup * 80

    let (posA, colA) = lattice(40) // 64000 pts
    let (posB, colB) = lattice(32) // 32768 pts
    let cpuA = PackedPointCloudFixtures.pack(positions: posA, colors: colA)
    let cpuB = PackedPointCloudFixtures.pack(positions: posB, colors: colB)

    let cloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(maxResidentBatches: cpuA.batchCount + cpuB.batchCount + 3, pointsPerBatch: ppb)
    )

    // First chunk -> slots [0, nbA).
    let sA = cloud.addRawChunksGPU(packer: packer, queue: queue,
                                   chunks: [(sharedBuffer(device, posA), sharedBuffer(device, colA), posA.count)])
    #expect(sA == [0])
    #expect(cloud.residentBatchCount == cpuA.batchCount)
    #expect(cloud.pointCount == posA.count)
    verifyChunkAtSlot(cloud, cpuA, firstSlot: 0, ppb: ppb, label: "A(after A)")

    // Second chunk, separate dispatch -> slots [nbA, nbA+nbB). Must NOT disturb A.
    let sB = cloud.addRawChunksGPU(packer: packer, queue: queue,
                                   chunks: [(sharedBuffer(device, posB), sharedBuffer(device, colB), posB.count)])
    #expect(sB == [cpuA.batchCount]) // non-zero firstSlot
    #expect(cloud.residentBatchCount == cpuA.batchCount + cpuB.batchCount)
    #expect(cloud.pointCount == posA.count + posB.count)
    verifyChunkAtSlot(cloud, cpuB, firstSlot: cpuA.batchCount, ppb: ppb, label: "B")
    verifyChunkAtSlot(cloud, cpuA, firstSlot: 0, ppb: ppb, label: "A(after B)") // undisturbed
}

@Test func gpuPackTwoChunksOneCommandBuffer() throws {
    let context = try makeContext()
    let device = context.device
    let queue = try #require(device.makeCommandQueue())
    let packer = try GPUPacker(context: context)
    let ppb = computeRasteriserThreadsPerGroup * 80

    let (posA, colA) = lattice(36) // 46656 pts
    let (posB, colB) = lattice(28) // 21952 pts
    let cpuA = PackedPointCloudFixtures.pack(positions: posA, colors: colA)
    let cpuB = PackedPointCloudFixtures.pack(positions: posB, colors: colB)

    let cloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(maxResidentBatches: cpuA.batchCount + cpuB.batchCount + 1, pointsPerBatch: ppb)
    )

    // Both chunks in ONE addRawChunksGPU call (one command buffer, shared scratch).
    let slots = cloud.addRawChunksGPU(packer: packer, queue: queue, chunks: [
        (sharedBuffer(device, posA), sharedBuffer(device, colA), posA.count),
        (sharedBuffer(device, posB), sharedBuffer(device, colB), posB.count),
    ])
    #expect(slots == [0, cpuA.batchCount])
    #expect(cloud.residentBatchCount == cpuA.batchCount + cpuB.batchCount)
    verifyChunkAtSlot(cloud, cpuA, firstSlot: 0, ppb: ppb, label: "A")
    verifyChunkAtSlot(cloud, cpuB, firstSlot: cpuA.batchCount, ppb: ppb, label: "B")
}
#endif
