import Foundation
import Metal
import Satin
import simd
import Testing
import SwiftPDAL
@testable import SatinComputeRasteriser
@testable import SatinComputeRasteriserStreaming

// MARK: - Env-gated integration harness against a real COPC dataset.
//
// Skips unless `SATIN_CR_STREAMING_TEST_COPC` names a readable COPC LAZ.
// Run:
//   SATIN_CR_STREAMING_TEST_COPC=/path/to/file.copc.laz \
//     swift test --filter streamingResidencyStaysConsistentWithSource
//
// The harness drives a scripted camera through a deliberately *undersized*
// slot pool so pool pressure is guaranteed, and checks that the adapter's
// GPU residency never permanently diverges from what the SwiftPDAL source
// believes is resident. It also records chunk-arrival ordering discipline
// and spot-checks decoded chunk data on real points.

// MARK: - Recording source wrapper (shadow-residency spy)

/// Forwards every `StreamingPointCloudSource` call to a real source, but
/// snoops each drained `StreamingUpdate` so the test can maintain a shadow
/// of what the *source* believes is GPU-resident (`believedResident`) and
/// run per-chunk data sanity as chunks arrive. Interposing here — between
/// the driver's publish queue and the adapter — is the only place the raw
/// deltas are visible, since the adapter owns the polling.
private final class RecordingStreamingSource: StreamingPointCloudSource, @unchecked Sendable {
    let inner: any StreamingPointCloudSource
    private let lock = NSLock()

    // Shadow of the source's residency model: every id ever in `added`
    // minus every id ever in `removed`.
    private(set) var believedResident: Set<ChunkID> = []
    private(set) var everAdded: Set<ChunkID> = []
    private(set) var arrivalOrder: [ChunkID] = []
    private(set) var totalAdded = 0
    private(set) var totalRemoved = 0

    // Data-sanity accumulators.
    private(set) var chunksChecked = 0
    private(set) var sanityFailures: [String] = []
    // Arrival-order diagnostics.
    private(set) var descendantBeforeParent = 0

    private let lodLevels = 8

    init(_ inner: any StreamingPointCloudSource) { self.inner = inner }

    var info: StreamingSourceInfo { inner.info }
    func submit(view: StreamingCameraView) { inner.submit(view: view) }
    func setBudget(_ bytes: Int) { inner.setBudget(bytes) }
    func nextUpdate() async -> StreamingUpdate? { await inner.nextUpdate() }
    func cancel(_ chunkIDs: [ChunkID]) { inner.cancel(chunkIDs) }
    func close() { inner.close() }

    func pollLatest() -> StreamingUpdate? {
        guard let delta = inner.pollLatest() else { return nil }
        lock.withLock {
            for id in delta.removed {
                believedResident.remove(id)
                totalRemoved += 1
            }
            for chunk in delta.added {
                // Parent-before-descendant discipline (report only; SwiftPDAL
                // documents no ordering guarantee). COPC parent of (d,x,y,z)
                // is (d-1, x/2, y/2, z/2).
                if chunk.id.depth > 0 {
                    let parent = ChunkID(depth: chunk.id.depth - 1,
                                         x: chunk.id.x / 2,
                                         y: chunk.id.y / 2,
                                         z: chunk.id.z / 2)
                    if !everAdded.contains(parent) { descendantBeforeParent += 1 }
                }
                believedResident.insert(chunk.id)
                everAdded.insert(chunk.id)
                arrivalOrder.append(chunk.id)
                totalAdded += 1
                checkChunk(chunk)
            }
        }
        return delta
    }

    // MARK: per-chunk data sanity (spot-check; this runs on a 13GB dataset)

    private func note(_ msg: String) {
        if sanityFailures.count < 20 { sanityFailures.append(msg) }
    }

    private func checkChunk(_ chunk: ResidentChunk) {
        chunksChecked += 1
        let id = chunk.id
        let fileMin = info.bounds.min - SIMD3<Float>(info.originShift)
        let fileMax = info.bounds.max - SIMD3<Float>(info.originShift)

        // Buffer-length invariants.
        let total = chunk.totalPointCount
        let want4 = total * 4
        if chunk.xyzLow.count != want4 || chunk.xyzMed.count != want4
            || chunk.xyzHigh.count != want4 || chunk.colors.count != want4 {
            note("\(id): position/color buffer length mismatch (points=\(total))")
        }
        if chunk.levels.count != total {
            note("\(id): levels length \(chunk.levels.count) != points \(total)")
        }

        let levels = [UInt8](chunk.levels)
        let low  = chunk.xyzLow.toUInt32()
        let med  = chunk.xyzMed.toUInt32()
        let high = chunk.xyzHigh.toUInt32()
        guard low.count == total, med.count == total, high.count == total,
              levels.count == total else {
            note("\(id): decoded array count mismatch"); return
        }

        var runningFirst = 0
        for (bi, b) in chunk.batches.enumerated() {
            let first = Int(b.firstPoint)
            let n = Int(b.numPoints)
            if first != runningFirst {
                note("\(id) batch \(bi): firstPoint \(first) != expected \(runningFirst)")
            }
            runningFirst += n
            if first + n > total {
                note("\(id) batch \(bi): slice [\(first),\(first+n)) exceeds \(total)"); continue
            }

            // Batch AABB should sit inside the file bounds (epsilon slack).
            let eps: Float = max(1.0, simd_length(fileMax - fileMin) * 1e-3)
            let bmin = SIMD3<Float>(b.minX, b.minY, b.minZ)
            let bmax = SIMD3<Float>(b.maxX, b.maxY, b.maxZ)
            if any(bmin .< (fileMin - eps)) || any(bmax .> (fileMax + eps)) || any(bmax .< bmin) {
                note("\(id) batch \(bi): AABB \(bmin)…\(bmax) outside file bounds \(fileMin)…\(fileMax)")
            }

            // Levels within [0, lodLevels) and non-decreasing across the slice.
            var prev: UInt8 = 0
            var mono = true
            var recount = [Int](repeating: 0, count: lodLevels)
            for i in first ..< first + n {
                let lv = levels[i]
                if Int(lv) >= lodLevels { note("\(id) batch \(bi): level \(lv) >= \(lodLevels)"); break }
                if lv < prev { mono = false }
                prev = lv
                recount[Int(lv) & (lodLevels - 1)] += 1
            }
            if !mono { note("\(id) batch \(bi): levels not non-decreasing") }

            // Cumulative LOD counts (padding3..6) must match a recount, and
            // cum[7] == numPoints. A zeroed cum7 is the "unbucketed" sentinel.
            for l in 1 ..< lodLevels { recount[l] += recount[l - 1] }
            // Cumulative counts are packed two-per-word (low level in low half)
            // across padding3..6 on the StreamingRasterBatch.
            let cum = [
                Int(b.padding3 & 0xFFFF), Int(b.padding3 >> 16),
                Int(b.padding4 & 0xFFFF), Int(b.padding4 >> 16),
                Int(b.padding5 & 0xFFFF), Int(b.padding5 >> 16),
                Int(b.padding6 & 0xFFFF), Int(b.padding6 >> 16),
            ]
            if cum[lodLevels - 1] != n {
                note("\(id) batch \(bi): cum7 \(cum[lodLevels-1]) != numPoints \(n) (LOD counts dropped?)")
            } else if cum != recount {
                note("\(id) batch \(bi): cum \(cum) != recount \(recount)")
            }

            // xyz decode spot-check: sample up to 8 points, decode the 30-bit
            // fixed-point back to world and confirm it lands inside the AABB.
            let step = max(1, n / 8)
            var s = first
            while s < first + n {
                let px = decodeAxis(low[s], med[s], high[s], axis: 0, min: b.minX, max: b.maxX)
                let py = decodeAxis(low[s], med[s], high[s], axis: 1, min: b.minY, max: b.maxY)
                let pz = decodeAxis(low[s], med[s], high[s], axis: 2, min: b.minZ, max: b.maxZ)
                let p = SIMD3<Float>(px, py, pz)
                let slack = simd_length(bmax - bmin) * 1e-3 + 1e-3
                if any(p .< (bmin - slack)) || any(p .> (bmax + slack)) || px.isNaN || py.isNaN || pz.isNaN {
                    note("\(id) batch \(bi): point \(s) decoded \(p) outside batch AABB \(bmin)…\(bmax)")
                    break
                }
                s += step
            }
        }
    }

    /// Reconstruct one axis from the three 10-bit fragments. Mirrors
    /// ChunkPacker: `qLow`=high 10 bits, `qMed`=mid, `qHigh`=low 10 bits;
    /// normalized = q / (2^30 - 1); world = min + normalized*(max-min).
    private func decodeAxis(_ lo: UInt32, _ me: UInt32, _ hi: UInt32,
                            axis: Int, min: Float, max: Float) -> Float {
        let sh = UInt32(axis * 10)
        let mask: UInt32 = 0x3FF
        let qL = (lo >> sh) & mask
        let qM = (me >> sh) & mask
        let qH = (hi >> sh) & mask
        let q = (qL << 20) | (qM << 10) | qH
        let norm = Float(q) / Float((1 << 30) - 1)
        return min + norm * (max - min)
    }
}

private extension Data {
    func toUInt32() -> [UInt32] {
        let n = count / 4
        var out = [UInt32](repeating: 0, count: n)
        _ = out.withUnsafeMutableBytes { dst in
            self.copyBytes(to: dst.bindMemory(to: UInt32.self), count: n * 4)
        }
        return out
    }
}

// MARK: - Per-tick record

private struct TickRecord {
    let tick: Int
    let phase: String
    let believed: Int
    let resident: Int
    let pending: Int
    let freeSlots: Int
    let cloudPoints: Int
    let residentPoints: Int
    let poolFull: Bool
    /// Chunks the source thinks are on the GPU that the adapter neither
    /// holds resident nor has queued for upload — the permanent-loss metric.
    var leak: Int { believed - resident - pending }
}

// MARK: - The test

@Test func streamingResidencyStaysConsistentWithSource() async throws {
    guard let path = ProcessInfo.processInfo.environment["SATIN_CR_STREAMING_TEST_COPC"],
          !path.isEmpty else {
        // Clean, explicit skip when the dataset isn't provided: no expectations
        // run, so the test passes as a no-op with a note in the log.
        print("[SKIP] streamingResidencyStaysConsistentWithSource: set SATIN_CR_STREAMING_TEST_COPC to a COPC path to run this harness.")
        return
    }
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: url.path) else {
        Issue.record("SATIN_CR_STREAMING_TEST_COPC points to a missing file: \(path)")
        return
    }

    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)

    // Open exactly as the app does, but with a fast tick.
    let cores = max(2, ProcessInfo.processInfo.activeProcessorCount)
    // distanceOnly scores *every* chunk and fills the byte budget with the
    // nearest ones — so the wanted set actually grows to the budget, unlike
    // frustumFirstThenHalo which only wants a thin LOD shell.
    let opts = StreamingOptions(
        maxInFlightLoads: cores * 2,
        decodeConcurrency: cores,
        driverTickInterval: .milliseconds(16),
        residencyPolicy: .distanceOnly
    )
    let real = try await CopcStreamingPointCloudSource.open(url, options: opts)
    let spy = RecordingStreamingSource(real)

    // Deliberately over-subscribed: the source's byte budget is far larger than
    // the pool's slot capacity, so the residency scorer keeps wanting (and the
    // driver keeps admitting) more chunks than the pool can hold. Decode
    // throughput — not the byte budget — bounds how many chunks actually arrive
    // in the test window (a few dozen), and each coarse COPC node spans many
    // 10 240-point batches, so a tiny ~64-slot pool is reliably overwhelmed by
    // the slot demand of even ~30 admitted chunks. This is the pool-full
    // pressure that surfaces the drop path.
    let pointsPerBatch = spy.info.pointsPerBatch
    let bytesPerSlot = pointsPerBatch * spy.info.bytesPerPoint
    let slots = 64
    let poolBytes = slots * bytesPerSlot
    let sourceBudget = 512 * 1024 * 1024
    let cap = ComputeRasteriserCapacity(maxResidentBatches: slots, pointsPerBatch: pointsPerBatch)
    let cloud = ComputeRasteriserPointCloud(context: context, capacity: cap,
                                            label: "StreamingHarness")
    spy.setBudget(sourceBudget)

    let adapter = StreamingAdapter(source: spy, cloud: cloud)
    // Drain aggressively so the pool actually fills and pressure shows up fast.
    adapter.maxChunkUploadsPerTick = 32

    let originShift = SIMD3<Float>(spy.info.originShift)
    let bmin = spy.info.bounds.min - originShift
    let bmax = spy.info.bounds.max - originShift
    let center = (bmin + bmax) * 0.5
    let radius = max(simd_length(bmax - bmin) * 0.5, 0.01)

    let camera = PerspectiveCamera(context: context, label: "harness",
                                   position: .zero, near: 0.01, far: 1000, fov: 45)
    camera.aspect = 16.0 / 9.0
    let viewport = SIMD2<Float>(1920, 1080)

    func place(position: SIMD3<Float>, target: SIMD3<Float>) {
        let d = max(simd_length(position - target), 0.01)
        camera.near = max(d - radius * 3, 0.001)
        camera.far = d + radius * 4
        camera.position = position
        camera.lookAt(target: target)
    }

    var records: [TickRecord] = []
    var maxFreed = 0
    var minFreeSlots = Int.max

    // One tick: update camera view, poll+apply delta, let the driver decode a
    // bit, then snapshot the invariants.
    func tick(_ phase: String, _ i: Int) async {
        adapter.update(camera: camera, viewport: viewport)
        // Give the background driver + decoders real time to produce chunks.
        try? await Task.sleep(for: .milliseconds(70))
        let rec = TickRecord(
            tick: records.count,
            phase: phase,
            believed: spy.believedResident.count,
            resident: adapter.residentChunks,
            pending: adapter.pendingUploadCount,
            freeSlots: cloud.freeSlotCount,
            cloudPoints: cloud.pointCount,
            residentPoints: adapter.residentPoints,
            poolFull: adapter.lastError != nil
        )
        records.append(rec)
        minFreeSlots = min(minFreeSlots, rec.freeSlots)
        maxFreed = max(maxFreed, rec.freeSlots)
    }

    let N = 18
    // Phase 1 — FAR: whole bounds.
    let farDist = radius / max(tanf(camera.fov * 0.5 * .pi / 180), 0.001) * 1.35
    let farPos = center + SIMD3<Float>(0, 0, farDist)
    for i in 0 ..< N { place(position: farPos, target: center); await tick("far", i) }

    // Phase 2 — ZOOM to a corner.
    let corner = bmax
    let zoomPos = corner + simd_normalize(corner - center) * (radius * 0.15) + SIMD3<Float>(0, 0, radius * 0.15)
    for i in 0 ..< N { place(position: zoomPos, target: corner); await tick("zoom", i) }

    // Phase 3 — ORBIT around the centre at mid radius.
    for i in 0 ..< N {
        let a = Float(i) / Float(N) * 2 * .pi
        let orbitPos = center + SIMD3<Float>(cosf(a), 0.3, sinf(a)) * (radius * 0.9)
        place(position: orbitPos, target: center)
        await tick("orbit", i)
    }

    // Phase 4 — HOLD STILL at the last orbit pose.
    let holdStart = records.count
    for i in 0 ..< N { await tick("hold", i) }

    adapter.close()

    // MARK: report

    func fmt(_ r: TickRecord) -> String {
        "  t\(r.tick) [\(r.phase)] believed=\(r.believed) resident=\(r.resident) pending=\(r.pending) leak=\(r.leak) free=\(r.freeSlots) full=\(r.poolFull ? "Y" : "-") pts(cloud=\(r.cloudPoints),adapter=\(r.residentPoints))"
    }
    let maxLeak = records.map(\.leak).max() ?? 0
    let leakTicks = records.filter { $0.leak > 0 }
    let maxBelieved = records.map(\.believed).max() ?? 0
    let maxPending = records.map(\.pending).max() ?? 0
    let anyPoolFull = records.contains { $0.poolFull }
    let holdRecords = Array(records[holdStart...])
    let holdLeakStart = holdRecords.first?.leak ?? 0
    let holdLeakEnd = holdRecords.last?.leak ?? 0
    // Recovery evidence: a retained (pending) chunk was flushed into a slot
    // freed by a later eviction — i.e. pending fell after having risen.
    var pendingRoseThenFell = false
    var sawPending = false
    for r in records {
        if r.pending > 0 { sawPending = true }
        else if sawPending { pendingRoseThenFell = true }
    }

    print("=== Streaming residency harness ===")
    print("file: \(url.lastPathComponent)  totalPoints=\(spy.info.totalPoints)  maxDepth=\(spy.info.maxDepth)")
    print("pool: \(slots) slots (\(poolBytes / (1024*1024)) MB, no 2x headroom)  sourceBudget=\(sourceBudget/(1024*1024)) MB  pointsPerBatch=\(pointsPerBatch)")
    print("source totals: added=\(spy.totalAdded) removed=\(spy.totalRemoved) everAdded=\(spy.everAdded.count)")
    print("pressure: maxBelieved=\(maxBelieved) vs poolSlots=\(slots)  maxPending=\(maxPending)  anyPoolFull=\(anyPoolFull)  pendingRecovered=\(pendingRoseThenFell)")
    print("chunksChecked=\(spy.chunksChecked) sanityFailures=\(spy.sanityFailures.count)")
    for f in spy.sanityFailures.prefix(20) { print("  SANITY: \(f)") }
    // Arrival order.
    let firstDepths = spy.arrivalOrder.prefix(24).map(\.depth)
    print("first 24 arrival depths: \(firstDepths)")
    print("descendant-before-parent arrivals: \(spy.descendantBeforeParent) of \(spy.totalAdded)")
    print("leak: max=\(maxLeak) leakTicks=\(leakTicks.count)  hold leak: start=\(holdLeakStart) end=\(holdLeakEnd)")
    print("freeSlots: min=\(minFreeSlots) max=\(maxFreed)")
    print("per-tick:")
    for r in records { print(fmt(r)) }

    // MARK: assertions (post-fix behaviour)

    // Pressure must actually have happened, or the test proves nothing: the
    // source wanted more chunks than the pool can physically hold.
    #expect(maxBelieved > slots || anyPoolFull,
            "pool pressure never reproduced (maxBelieved=\(maxBelieved) <= slots=\(slots), no pool-full)")

    // (5) adapter's residentPoints mirrors the cloud every tick.
    for r in records {
        #expect(r.residentPoints == r.cloudPoints,
                "tick \(r.tick): residentPoints \(r.residentPoints) != cloud.pointCount \(r.cloudPoints)")
    }
    // (1)+(2) No permanent divergence: the source's believed-resident set must
    // always be covered by resident + still-queued chunks. Pre-fix the pool-full
    // drop consumes and discards the chunk, so leak>0 and never heals.
    #expect(maxLeak == 0, "residency leak detected (max=\(maxLeak)); pool-full drop lost chunks the source still believes resident")
    #expect(holdLeakEnd == 0, "hold-still phase did not heal: leak still \(holdLeakEnd)")
    // (4) Real-data sanity must be clean.
    #expect(spy.sanityFailures.isEmpty, "per-chunk data sanity failures: \(spy.sanityFailures.prefix(5))")
    // (5) Slots were reused: evictions freed slots that later admits consumed.
    #expect(spy.totalRemoved > 0, "expected evictions under pressure")
}
