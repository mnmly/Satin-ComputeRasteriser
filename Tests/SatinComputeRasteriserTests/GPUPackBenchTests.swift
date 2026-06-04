#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// De-risk benchmark for the "offload streaming pack/quantize to the GPU" idea.
//
// The streaming path CPU-packs each COPC node (~140k points) on a worker
// thread via SwiftPDAL's ChunkPacker. GPU-offloading that means a per-node
// dispatch: upload raw points, run the pack kernels, sync back. The open
// question is whether the per-node kernel-launch + synchronisation overhead
// erases the win — unified memory removes the *copy* cost on Apple Silicon,
// but not the launch/sync cost.
//
// GPUPacker fed a single node's points is a faithful proxy for the per-chunk
// kernel (global bounds == node bounds when only that node is present), and
// `replacePackedPointCloud` does exactly the realistic per-node work:
// encode → commit → waitUntilCompleted. We compare its wall time against the
// CPU pack, and also measure GPU throughput when many nodes are packed in one
// dispatch (amortising the launch). The gap between the two tells us how much
// of the GPU cost is fixed per-dispatch overhead.
//
// Run for real numbers (Debug is retain/release noise):
//   swift test -c release --filter PackBench

private func makeContext() throws -> Context {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    return Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
}

private func sharedBuffer<T>(_ device: MTLDevice, _ array: [T]) -> MTLBuffer {
    array.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}

// A clustered, non-lattice point set roughly resembling a real COPC node:
// several gaussian-ish blobs inside a unit box. Deterministic (seeded LCG) so
// runs are comparable.
private func nodeLikePoints(_ count: Int) -> ([SIMD3<Float>], [SIMD4<Float>]) {
    var state: UInt64 = 0x9E3779B97F4A7C15
    func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF) // [0,1)
    }
    let blobCount = 24
    var centers: [SIMD3<Float>] = []
    for _ in 0 ..< blobCount {
        centers.append(SIMD3<Float>(next(), next(), next()))
    }
    var positions = [SIMD3<Float>](); positions.reserveCapacity(count)
    var colors = [SIMD4<Float>](); colors.reserveCapacity(count)
    for i in 0 ..< count {
        let c = centers[i % blobCount]
        let jitter = SIMD3<Float>(next() - 0.5, next() - 0.5, next() - 0.5) * 0.12
        let p = simd_clamp(c + jitter, .zero, SIMD3<Float>(repeating: 1)) - SIMD3<Float>(repeating: 0.5)
        positions.append(p)
        colors.append(SIMD4<Float>(next(), next(), next(), 1))
    }
    return (positions, colors)
}

private func medianMS(_ samples: [Double]) -> Double {
    let s = samples.sorted()
    return s[s.count / 2]
}

private func timeMS(_ iterations: Int, _ body: () -> Void) -> Double {
    var samples: [Double] = []
    for _ in 0 ..< iterations {
        let t0 = DispatchTime.now().uptimeNanoseconds
        body()
        let t1 = DispatchTime.now().uptimeNanoseconds
        samples.append(Double(t1 - t0) / 1_000_000.0)
    }
    return medianMS(samples)
}

@Test func packBenchGPUvsCPUPerNode() throws {
    let context = try makeContext()
    let device = context.device
    let queue = try #require(device.makeCommandQueue())
    let packer = try GPUPacker(context: context)

    let nodeCount = 140_000          // ~ a real COPC node
    let (positions, colors) = nodeLikePoints(nodeCount)
    let ppb = computeRasteriserThreadsPerGroup * 80

    // --- CPU pack (Satin's pack; same shape of work as SwiftPDAL ChunkPacker) ---
    // Warm up, then time.
    _ = PackedPointCloudFixtures.pack(positions: positions, colors: colors)
    let cpuMS = timeMS(30) {
        _ = PackedPointCloudFixtures.pack(positions: positions, colors: colors)
    }

    // --- GPU pack, single node, realistic per-node cost (encode+commit+wait) ---
    let cloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(
            maxResidentBatches: (nodeCount / ppb) + 4, pointsPerBatch: ppb
        )
    )
    let posBuf = sharedBuffer(device, positions)
    let colBuf = sharedBuffer(device, colors)
    // Warm up: compiles pipelines + grows pool/scratch (one-time, not per-node).
    cloud.replacePackedPointCloud(packer: packer, queue: queue, positions: posBuf, colors: colBuf, count: nodeCount)
    let gpuPerNodeMS = timeMS(30) {
        cloud.replacePackedPointCloud(packer: packer, queue: queue, positions: posBuf, colors: colBuf, count: nodeCount)
    }

    // --- GPU pack, many nodes in ONE dispatch (amortises launch/sync) ---
    let batched = 10
    let (bigPos, bigCol) = nodeLikePoints(nodeCount * batched)
    let bigCloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(
            maxResidentBatches: (nodeCount * batched / ppb) + 8, pointsPerBatch: ppb
        )
    )
    let bigPosBuf = sharedBuffer(device, bigPos)
    let bigColBuf = sharedBuffer(device, bigCol)
    bigCloud.replacePackedPointCloud(packer: packer, queue: queue, positions: bigPosBuf, colors: bigColBuf, count: nodeCount * batched)
    let gpuBatchedMS = timeMS(15) {
        bigCloud.replacePackedPointCloud(packer: packer, queue: queue, positions: bigPosBuf, colors: bigColBuf, count: nodeCount * batched)
    }
    let gpuAmortizedPerNodeMS = gpuBatchedMS / Double(batched)
    let overheadMS = gpuPerNodeMS - gpuAmortizedPerNodeMS

    print("""

    ===================== PACK BENCH (median, ~\(nodeCount) pts/node) =====================
    CPU pack / node ...................... \(String(format: "%.3f", cpuMS)) ms
    GPU pack / node (encode+commit+wait) . \(String(format: "%.3f", gpuPerNodeMS)) ms
    GPU pack / node (amortised x\(batched)) ...... \(String(format: "%.3f", gpuAmortizedPerNodeMS)) ms
    -> per-dispatch overhead (sync etc.) . \(String(format: "%.3f", overheadMS)) ms
    -------------------------------------------------------------------------
    per-node speedup (CPU / GPU-with-sync) \(String(format: "%.2fx", cpuMS / gpuPerNodeMS))
    amortised speedup (CPU / GPU-amortised) \(String(format: "%.2fx", cpuMS / gpuAmortizedPerNodeMS))
    VERDICT(per-node):  \(cpuMS > gpuPerNodeMS ? "GPU faster even per-node" : "CPU faster per-node — dispatch overhead dominates")
    =========================================================================

    """)

    // Not an assertion on perf (machine-dependent) — just guard the bench ran.
    #expect(cpuMS > 0 && gpuPerNodeMS > 0 && gpuBatchedMS > 0)
}

// The REAL streaming primitive: addRawChunksGPU packs a frame's worth of nodes
// into free pool slots in one command buffer (contiguous reservation + per-chunk
// pack + finalize + mirror flush). Unlike the wholesale proxy above, the chunks
// serialise on shared sort scratch (Metal hazard tracking), so this is the
// honest per-frame cost of the Phase-1 integration primitive.
@Test func packBenchAddRawChunksGPUPerFrame() throws {
    let context = try makeContext()
    let device = context.device
    let queue = try #require(device.makeCommandQueue())
    let packer = try GPUPacker(context: context)
    let ppb = computeRasteriserThreadsPerGroup * 80

    let nodeCount = 140_000
    let nodesPerFrame = 8                          // typical streaming delta.added
    let (positions, colors) = nodeLikePoints(nodeCount)
    let posBuf = sharedBuffer(device, positions)
    let colBuf = sharedBuffer(device, colors)
    let nb = (nodeCount + ppb - 1) / ppb
    let cloud = ComputeRasteriserPointCloud(
        context: context,
        capacity: ComputeRasteriserCapacity(maxResidentBatches: nb * nodesPerFrame + 4, pointsPerBatch: ppb)
    )
    let frameChunks = Array(repeating: (positions: posBuf, colors: colBuf, count: nodeCount), count: nodesPerFrame)

    // CPU: pack the same N nodes (what the worker threads do today).
    _ = PackedPointCloudFixtures.pack(positions: positions, colors: colors)
    let cpuFrameMS = timeMS(20) {
        for _ in 0 ..< nodesPerFrame { _ = PackedPointCloudFixtures.pack(positions: positions, colors: colors) }
    }

    // GPU: one addRawChunksGPU dispatch per frame, sweeping the scratch-ring
    // width so we see how much per-chunk scratch (GPU overlap) helps. Clear
    // between iterations so slots are free again (cheap; not the hot path).
    let cpuPerNode = cpuFrameMS / Double(nodesPerFrame)
    var lines: [String] = []
    for sets in [1, 2, 4, 8] {
        cloud.addRawChunksGPU(packer: packer, queue: queue, chunks: frameChunks, maxConcurrent: sets) // warm
        cloud.clearAllBatches()
        let ms = timeMS(20) {
            cloud.addRawChunksGPU(packer: packer, queue: queue, chunks: frameChunks, maxConcurrent: sets)
            cloud.clearAllBatches()
        }
        lines.append(String(format: "  scratch sets = %d ... %7.3f ms/frame  (%.3f ms/node)  %.2fx vs CPU",
                            sets, ms, ms / Double(nodesPerFrame), cpuFrameMS / ms))
    }

    print("""

    ============ PACK BENCH — REAL PRIMITIVE (addRawChunksGPU, \(nodesPerFrame) nodes/frame) ============
    CPU pack \(nodesPerFrame) nodes/frame ........ \(String(format: "%.3f", cpuFrameMS)) ms   (\(String(format: "%.3f", cpuPerNode)) ms/node)
    \(lines.joined(separator: "\n"))
    =========================================================================

    """)
    #expect(cpuFrameMS > 0)
}
#endif
