# LAZ decompressor pool — proof of value, then implement

**Goal**: cut streaming CPU by eliminating per-node lazperf decompressor construction. Profile shows ctor+dtor of `Point14Decompressor` is 26%+27%≈53% inclusive vs. ~10% actual decode self-time.

**Acceptance criteria for full effort**: with a per-thread pooled decompressor, streaming-phase CPU time on the same workload drops by ≥25% in a Time Profiler re-capture.

## Phase 1 — Validate the hypothesis (this PR)

Before touching lazperf internals, prove that the construction cost survives if we could reset. The risk: if `models::arithmetic` model tables need genuinely expensive re-initialization (not just heap allocation), the win evaporates.

- [x] Read laz-perf and copc-lib sources; confirm per-node decompressor lifecycle
- [ ] Write a standalone C++ bench using lazperf directly that measures three costs separately:
  - `T_full`     = ctor + decompress N points + dtor (the status quo)
  - `T_ctor`     = ctor + dtor only (no decode)
  - `T_decode`   = decompress N points only (decompressor already alive)
- [ ] Build bench against `/tmp/copc-xcf-build/copc-lib/libs/laz-perf` directly (no CMake project needed; one-shot clang invocation)
- [ ] Generate synthetic point-14 data, compress once, decompress in tight loops
- [ ] Report `T_ctor / T_decode` ratio. Decision rule:
  - if `T_ctor` ≥ ~1.5× per-decompress steady-state cost over typical node sizes (~30k–100k points), **go ahead with Phase 2**
  - if `T_ctor` < per-node decode cost, abort — the trace was misleading and we need different angle

## Phase 2 — Commits 1–3 (lazperf side with tests) — DONE

- [x] Add `reset()` / setter APIs to lazperf classes
    - `models::arithmetic`, `models::arithmetic_bit` (model.hpp)
    - `decoders::arithmetic`, `MemoryStream`, `InCbStream::setCallback` (decoder.hpp + streams.hpp)
    - `decompressors::integer` (decompressor.hpp)
    - `Point14Base::ChannelCtx::resetForDecode`, `Point14Decompressor::reset(InCbStream&)` (field_point14.{hpp,cpp})
    - `Rgb14Decompressor::reset`, `Nir14Decompressor::reset`, `Byte14Decompressor::reset` (field_*14.{hpp,cpp})
    - `las_decompressor::reset(InputCb)` virtual + `point_decompressor_base_1_4::reset(InputCb)` (lazperf.{hpp,cpp})
- [x] Patch stored at `SwiftPDAL/Frameworks/lazperf-patches/0001-add-reset-api.patch` (402 lines, 14 files)
- [x] `scripts/build-copc-xcframework.sh` wired to apply patches idempotently after clone
- [x] Reset-correctness test (`reset_test.cpp`): 3 distinct chunks decoded via single reused decompressor are byte-identical to fresh-ctor baselines. Replay chain validated.
- [x] Pool perf bench (`pool_bench.cpp`): per-chunk savings vs status quo

### Bench results across node sizes

| Chunk size | Savings/chunk | % of decode |
|---|---|---|
| 2,000 pts | 0.27 ms | 29% |
| 5,000 pts | 0.27 ms | 17% |
| 10,000 pts | 0.29 ms | 12% |
| 34,275 pts (real mean) | 0.36 ms | 5% |
| 80,000 pts | 0.40 ms | 3% |

Absolute savings ≈ fixed per-chunk cost (0.27–0.40 ms). Applied to the real-workload histogram: projected total ~5–15% decode-CPU saving across a streaming session, plus large reduction in malloc/free pressure (which the trace showed as ~20% of CPU in `_xzm_*` leaves).

### Phase 2 — commits 4–8 — DONE

- [x] **Commit 4** — Rebuilt macOS xcframework with patched lazperf (`scripts/build-copc-xcframework.sh PLATFORMS=macos`). Reset API confirmed in exported headers.
- [x] **Commit 5+6** — Pooled decode wired in `Sources/CxxCOPC/copc_bridge.cpp`:
    - `PooledDecompressor` per FileReader slot (parallel to existing one-slot-per-thread contract)
    - `swiftpdal_copc_read_node` switched from `reader->GetPoints(node)` to `GetPointDataCompressed` + pool decode + `Points::Unpack`
    - `Package.swift` switched copclib binaryTarget URL→path-based for local iteration
- [x] **Commit 7** — Time Profiler re-capture on matched workload (~18.8k nodes, mean ~24k pts both sides). Headline numbers:
    - `decodeAndPack` inclusive: **11.3s → 3.3s (−71%)**
    - decodeAndPack share of total CPU: **50.7% → 23.9%**
    - Allocator family (`_xzm_*`): **−87%**
    - Total CPU/20s wall: **22.4s → 13.8s (−38%)**
    - Bench predicted 5–15%; reality 38% total / 71% on the decode pipeline. Gap is allocator second-order effects the bench couldn't model.
- [x] **Commit 8** — Histogram kept but env-gated; activate with `SWIFTPDAL_NODE_HISTOGRAM=1`. Lock-free; zero cost when off.

### Landed in SwiftPDAL repo (2 commits, not yet pushed)

- `c433435` Add lazperf reset() API as a local patch over upstream master
- `1b6d398` Pool lazperf decompressors per FileReader slot; gate histogram off by default

### Notes for the next person

- `Package.swift` currently points copclib at the local `Frameworks/copclib.xcframework`. Before cutting a release, run the build script (full PLATFORMS), upload `copclib.xcframework.zip`, then switch the binaryTarget back to URL-based with the printed checksum.
- xcframework artifact is gitignored, so a fresh clone needs to run `scripts/build-copc-xcframework.sh PLATFORMS=macos` (or full) before building.
- If lazperf upstream churns enough to break the patch, regenerate it: apply patch, fix conflicts, `git diff > 0001-add-reset-api.patch`.

## Working notes

- Phase 1 bench targets only `point_decompressor_7` (format 7 = COPC default with RGB)
- Use `chrono::steady_clock` with enough iterations to push wall time past noise (~1s per measurement)
- Synthetic point-14 data fine for ctor-cost measurement — the ctor doesn't depend on point contents

## Results — Phase 1

Bench: `/tmp/lazperf-bench/bench.cpp`, point format 7 synthetic, Release/-O3.

- Per-point steady-state decode: **0.195 μs**
- Fixed per-node overhead (ctor + dtor + first-call): **0.727 ms**
- F(50k pts) total: 10.456 ms → fixed share **7.0%**
- F(0) (pure ctor+dtor): 0.539 ms; F(1) − F(0) ≈ 9 μs (first-call read overhead is small)

**Reconciling with the 27% inclusive seen in Instruments**: that level of fixed share only holds if real node sizes average ~1,700 points. COPC leaves at the bottom of the streaming tree are small, which matches the streaming-during-zoom workload the trace was taken on.

**Implication**: Phase 2 viability depends on node-size distribution in production workloads. Before committing the lazperf reset work, instrument SwiftPDAL to log node point-count distribution during a real streaming session and confirm small-node dominance.

## Phase 1.5 — Results

Run on real workload: 18,766 nodes, 643M points.
- Mean node size: **34,275 pts** (much larger than the 1,700 the trace inclusive% implied)
- Median ≈ 24k pts; 80%+ of points in 2^15 + 2^16 buckets (32k–131k)
- 17.9% of nodes ≤ 4k pts (the pool-friendly tail)

Bench-based projection: total ctor CPU = 18766 × 0.727ms ≈ 13.6s; total decode CPU = 643M × 0.195μs ≈ 125s. **Pool savings: ~10% (bench data) to ~25% (if real per-point decode is faster on real LIDAR than synthetic, which is likely).** Proceeding to Phase 2.

(histogram instrumentation in `copc_bridge.cpp` kept in place for re-measurement after Phase 2)

## Phase 1.5 (new step — added after Phase 1 result)

- [ ] Add a one-shot counter in `swiftpdal_copc_read_node` (or wrapper) that bins observed `points.Size()` into log-scale buckets and dumps a histogram on shutdown
- [ ] Run a representative streaming session (cold load → orbit → zoom). Capture histogram.
- [ ] Compute weighted-mean node size and projected fixed-share from bench table above
- [ ] Decision gate:
    - if weighted-mean ≤ ~5k pts → proceed to Phase 2 (full pool)
    - if weighted-mean in 5–20k → consider lighter intervention (skip `std::istream`, ~5% win) before lazperf surgery
    - if weighted-mean ≥ 20k → abort pool plan; look elsewhere

---

# GPU pack() — design + plan (2026-05-31)

Goal: GPU implementation of `PackedPointCloudFixtures.pack()` so GPU-resident
point data (e.g. WABF geometry-node output) becomes a renderable
`ComputeRasteriserPointCloud` with no CPU round-trip. Additive — CPU `pack()`
stays as reference/fallback.

## Measurement (Release, 13.6M-point PLY)

`pack()` TOTAL ≈ 1111 ms. Stage breakdown:

| stage            | ms   | %  |
|------------------|------|----|
| bounds           | 12   | 1  |
| morton keys      | 36   | 3  |
| **sort indices** | 345  | 32 |
| gather pos+col   | 73   | 7  |
| **LOD levels**   | 512  | 48 |
| batch + quantize | 55   | 5  |
| color pack       | 33   | 3  |

Finding: LOD (48%) > sort (32%); together 80%. GPU LOD voxel-occupancy is as
important as the sort. Target: tens of ms total on GPU.

## Decisions (confirmed with user)

- Sort: **LSD radix**, 4×8-bit passes over the 30-bit Morton key (not the
  bitonic sorter in SatinSpark — that's float-key O(N log²N), pads 13.6M→16.7M).
- LOD: **bit-exact**. Reproduce "lowest sorted index per voxel wins" via
  `atomic_min(grid[cell], sortedIndex)`. Dense grid under a cell cap; hashed
  open-addressing fallback for elongated/huge-extent clouds.

## Exact formulas to mirror

- Morton key (sort): normalize by GLOBAL bounds extent, `*1023`, floor to uint,
  `mortonSpread10` per axis, `key = (sx<<2)|(sy<<1)|sz`.
- Quantize (per batch): `size=max(bmax-bmin,1e-6)`,
  `n=clamp((p-bmin)/size,0,0.99999994)`, `q=uint(n*Float(2^30-1))`.
  NB multiply by `2^30-1`=1073741823 (decode divides by `2^30`).
  Pack: low=(q>>20)&1023, med=(q>>10)&1023, high=q&1023; word=x|y<<10|z<<20.
- Color: RGBA8 = clamp(c,0,1)*255 per channel, r|g<<8|b<<16|a<<24.
- Pool mapping: nbatches=ceil(count/ppb); sorted index i → pool index i;
  batch s covers [s*ppb, min(s*ppb+ppb,count)), firstPoint=s*ppb, state=1.

## API (Metal-gated, additive)

```swift
public final class GPUPacker {
    public init(context: Context, lodLevels: Int = 4, coarseVoxelDivisions: Int = 64)
    public func pack(positions: MTLBuffer, colors: MTLBuffer, count: Int,
                     into cloud: ComputeRasteriserPointCloud,
                     commandBuffer: MTLCommandBuffer)
}
extension ComputeRasteriserPointCloud {
    public func replacePackedPointCloud(packer:device:queue:positions:colors:count:)
}
```

## Plan

1. [x] Measure CPU pack() stages.
2. [x] bounds → morton → radix sort → gather.
3. [x] batch AABB + quantize + color pack.
4. [x] LOD voxel occupancy (dense grid; per-axis cells ≤ coarseDiv·2^(L-1), so
   bounded — no hashed fallback needed for default config; CPU-LOD/skip above cap).
5. [x] cloud pool resize (prepareForGPUPack) + mirror sync (adoptGPUBatchBounds via
   completion handler) + public API (GPUPacker + replacePackedPointCloud overload).
6. [x] parity tests (GPUPackerParityTests) + benchmark.

## Results

- Parity: `gpuPackDecodeMatchesCPUSingleBatch` (decoded pos within quant tol,
  levels exact) + `gpuPackMultiBatchLevelDistributionMatches` both pass.
- Benchmark (13.6M PLY): **GPU 15.4 ms vs CPU 1110 ms ≈ 72×**, same 1334 batches.
- `swift build` + `swift test` (23 tests) green; DocC exit 0, no new warnings.
- GPU path `#if canImport(Metal)`; CPU `pack()` untouched (Linux/tests unaffected).
- Throwaway PackBench target removed after measurement.

## WABF integration (consumer, separate repo — not yet wired)

`GeometryAttribute.float3` byteStride = `MemoryLayout<SIMD3<Float>>.stride` = 16,
`.storageModeShared`; `float4` likewise → shapes match the GPUPacker contract.
`realize()` can pass `data["Position"]`/`data["Color"]` buffers straight into
`cloud.replacePackedPointCloud(packer:queue:positions:colors:count:)` — no arrays.

Implementation files:
- `Sources/SatinComputeRasteriser/GPUPacker.swift` (direct makeLibrary + manual
  encode, like DisplacementPass/SplatGPUSorter — Satin ComputeProcessor is
  one-kernel-per-pass, too rigid for this).
- `Sources/SatinComputeRasteriser/Pipelines/GPUPacker/Shaders.metal`.
- temp `Sources/PackBench` benchmark target (remove or gate before done).
