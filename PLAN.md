# Satin Compute Rasteriser Port Plan

## Goal

Port the most recent practical path from `/Users/mnmly/Development-local/GitHub/cpp/compute_rasterizer` into a Satin/Metal Swift package in this directory.

The target is not a full historical port of every OpenGL/CUDA variant. The target is the current `master` checkout at commit `f2cbb65` (`enable multi-threaded loading; with warning that it is buggy`) and, within that, the batch-oriented LAS compute rasteriser path:

- Primary source: `modules/compute_loop_las_hqs`
- Loader source: `modules/compute/LasLoaderSparse.*`
- Useful earlier/simple references: `modules/compute_loop_las`, `modules/compute_loop_las2`
- Avoid for the first port: CUDA path, VR paths, Potree node LOD paths, OpenGL draw/debug UI variants

The package should use the same Satin dependency as `../Satin-Spark/Package.swift`:

```swift
.package(url: "https://github.com/mnmly/Satin", branch: "feature/2.0-shader-source-transforms")
```

## Existing References

Use these as local implementation references:

- Satin framework source: `../Satin`
- Similar Satin module: `../Satin-Spark`
- Existing attempted port: `/Users/mnmly/Development-local/Personal/WABF/Sources/WABFComputeRasteriser`
- Satin skill notes: `/Users/mnmly/.claude/skills/satin-3d/SKILL.md`

The WABF attempt already has the right Satin integration shape:

- `RasterisedPointCloud` as an `Object` that owns compute passes and an output texture
- `RasterisedPointCloudNode` for point cloud inputs
- `ComputeProcessor` subclasses for depth, color, clear, resolve, and post passes
- Satin `SourceMaterial`/`PostProcessor` compositing
- Correct handling for `Object.context` being unavailable until Satin setup

The C++ source contains the algorithm we actually want:

- Packed 10/20/30-bit point coordinates split across `ssXyzLow`, `ssXyzMed`, `ssXyzHig`
- `Batch` metadata with min/max bounds, point count, first point, and file index
- Per-file transform metadata
- Per-batch frustum culling
- Adaptive precision based on projected batch size
- Depth pass writes closest depth per pixel
- Color pass averages points near the closest depth
- Resolve pass writes a display texture

Important Satin/Metal difference: Satin uses reverse Z. The C++ OpenGL path stores `floatBitsToInt(pos.w)` and uses `atomicMin` against a cleared `-Infinity` bit pattern. The Metal port should not copy that literally. Depth encoding, clear values, comparison direction, and resolve/debug checks need to be designed for Satin's reverse-Z convention.

## Target Package Shape

Create a Swift package named `Satin-ComputeRasteriser` with:

- `Sources/SatinComputeRasteriser`
- `Sources/SatinComputeRasteriser/Pipelines`
- `Sources/SatinComputeRasteriserDemo`
- `Tests/SatinComputeRasteriserTests`

Initial products:

- Library: `SatinComputeRasteriser`
- Demo executable: `satin-compute-rasteriser-demo`
- Optional fixture executable once tests need rendered image output: `satin-compute-rasteriser-fixture`

Initial dependencies:

- Satin from the same branch as Satin-Spark
- System `z` only if/when LAZ support or compressed parsing requires it

Do not copy WABF wholesale. Use it as an integration template and port the C++ batch data model and shader logic deliberately.

## Public API Sketch

Keep the first API narrow:

```swift
public final class ComputeRasteriser: Object {
    public var configuration: ComputeRasteriserConfiguration
    public func addPointCloud(_ cloud: ComputeRasteriserPointCloud)
    public func resize(size: (width: Float, height: Float), scaleFactor: Float)
    public func draw(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer)
}

public final class ComputeRasteriserPointCloud: Object {
    public init(context: Context, packed: PackedPointCloud)
}

public struct PackedPointCloud {
    public var batches: [RasterBatch]
    public var files: [RasterFile]
    public var xyzLow: [UInt32]
    public var xyzMed: [UInt32]
    public var xyzHigh: [UInt32]
    public var colors: [UInt32]
}
```

Add LAS/LAZ loaders behind this API after the renderer works with deterministic packed fixtures.

## Porting Phases

### Phase 1: Package and Satin Wiring

1. Create `Package.swift` with the Satin dependency branch from Satin-Spark.
2. Add the library target, demo target, and resource-copy rule for `Pipelines`.
3. Add a minimal demo renderer following `SatinSparkDemoRenderer` patterns:
   - `Renderer(context: defaultContext)`
   - `PerspectiveCamera`
   - `PerspectiveCameraController`
   - scene containing `ComputeRasteriser`
4. Verify `swift build` before shader work grows.

### Phase 2: Shared Types and Fixture Data

1. Define C/Metal-shared layout types in `Pipelines/ComputeRasteriserTypes.h`.
2. Mirror those types in Swift only where needed for `MemoryLayout` checks.
3. Start with a deterministic in-memory fixture:
   - small grid/cube of points
   - one `RasterFile`
   - one or more `RasterBatch`
   - packed `xyzLow/Med/High` buffers and packed `R | G << 8 | B << 16` colors
4. Add tests that assert Swift stride/layout matches shader expectations.

Important layout rule from the Satin notes: use SIMD types for `float3`/`uint3`-like fields or import shared C structs. Metal `float3`/`uint3` stride and alignment are not tuple-compatible.

### Phase 3: Metal Compute Passes

Port the C++ `compute_loop_las_hqs` shader path into Metal:

1. `ClearProcessor`
   - Clear depth buffer to the Metal equivalent of "empty".
   - Clear RGBA accumulation/count buffer.
2. `DepthPassProcessor`
   - Dispatch one workgroup per batch.
   - Use `local_size_x` equivalent of 128 threads where possible.
   - Reconstruct 10/20/30-bit coordinates from packed buffers.
   - Apply per-file transform.
   - Frustum-cull batches.
   - Compute adaptive precision level.
   - Atomic write closest depth per pixel using reverse-Z semantics.
3. `ColorPassProcessor`
   - Reconstruct the same point position.
   - Compare against the stored closest depth with a reverse-Z-aware version of the C++ `1%` HQS tolerance.
   - Accumulate RGB and contribution count.
   - For Metal portability, replace NVIDIA subgroup partitioned reductions with simple atomics first.
4. `ResolveProcessor`
   - Average accumulated color.
   - Write to a private `.rgba8Unorm` or `.bgra8Unorm` output texture.
5. Optional post pass:
   - Composite the output texture with a Satin `PostProcessor`, using the WABF post pattern.

Do not try to emulate NVIDIA-only subgroup extensions in the first pass. Correctness first; optimize after image parity is established.

### Phase 4: Renderer Object

Implement `ComputeRasteriser` as the orchestration layer:

1. Own per-frame pixel buffers:
   - depth buffer
   - color/count accumulation buffer
   - output texture
2. Own per-cloud GPU buffers:
   - batches
   - files
   - xyz low/med/high
   - colors
3. Resize buffers when viewport changes.
4. In `update(renderContext:camera:viewport:index:)`, push:
   - view matrix
   - projection matrix
   - view-projection matrix
   - image size
   - per-file transforms
5. In `encode(_:)`, run:
   - clear
   - depth pass per cloud
   - color pass per cloud
   - resolve
6. In `draw(...)`, composite the output texture after normal Satin scene rendering.

Follow the WABF deferred-context pattern: resize requests may arrive before Satin has assigned `context`.

### Phase 5: Loader Port

After the GPU path renders deterministic fixtures, port loading.

First loader target:

- LAS only
- Point formats with RGB offsets matching the C++ code:
  - format 2: RGB at byte 20
  - format 3: RGB at byte 28
  - format 7/8: RGB at byte 30

Later loader target:

- LAZ support, probably via a small C/C++ shim around LASzip if we want parity with `LasLoaderSparse.cpp`.

Loader output should be `PackedPointCloud`, not Metal buffers. GPU upload remains owned by the Satin object.

Port these C++ details carefully:

- File-local positions are shifted by `boxMin`.
- Batches are `POINTS_PER_WORKGROUP` points.
- Batch records are 64 bytes in the C++ path.
- Coordinates are normalised into 30-bit integer space per batch.
- The C++ shader has a suspicious typo in the 30-bit reconstruction path where Y/Z use `X_12`; decide whether to preserve for parity tests or correct it after confirming intended behavior.
- The C++ depth path is not directly portable because Satin uses reverse Z. Treat the C++ `atomicMin(floatBitsToInt(pos.w))` as algorithmic intent, not as the Metal implementation.

### Phase 6: Validation

Use progressive validation:

1. Unit tests for packing/unpacking:
   - encode known coordinates into 10/20/30-bit buffers
   - decode with a Swift helper matching the Metal shader
   - compare against original positions within quantisation error
2. Unit tests for batch bounds:
   - generated batches contain the expected min/max
   - empty or single-axis-flat batches avoid divide-by-zero issues
3. Shader fixture render:
   - render a deterministic point grid
   - verify non-empty output
   - compare selected pixel colors/depths
4. Demo validation:
   - load fixture
   - orbit camera
   - resize window
   - verify no Metal validation errors
5. Performance baseline:
   - compare raw float3 WABF-style path versus packed batch path
   - record points/sec and frame time for at least one synthetic point count

## Key Technical Decisions

- Use Satin `ComputeProcessor` subclasses for all compute kernels, matching the Satin skill guidance.
- Bind buffers through Satin custom indices; let Satin own the uniform buffer binding.
- Use Satin-compatible reverse-Z depth throughout the Metal port. Clear to the far/empty value, use the matching atomic direction, and keep color-pass tolerance comparisons in the same depth space.
- Prefer shared C header structs for shader/Swift layout consistency.
- Use private Metal buffers/textures for GPU-only frame data.
- Use shared or managed upload buffers only at load/build time.
- Keep the initial Metal shaders simple and portable. The C++ path depends on NVIDIA subgroup extensions that do not map directly to Apple GPUs.
- Render into an offscreen texture and composite with Satin, rather than trying to write directly into Satin's memoryless render target.

## Non-Goals for the First Implementation

- CUDA path
- VR paths
- Potree node LOD path
- Bounding-box debug draw
- Drag-and-drop UI
- Multi-threaded streaming loader
- LAZ support, unless LAS-only blocks useful testing
- Exact performance parity with RTX/NVIDIA subgroup implementation

## Immediate Next Steps

1. Scaffold the Swift package with the Satin dependency branch from Satin-Spark.
2. Copy/adapt the WABF processor structure into this package under the new names.
3. Add shared type definitions and deterministic packed fixture generation.
4. Implement clear, depth, color, and resolve kernels for packed batch data.
5. Build the demo and run `swift build`.
6. Add the first fixture test once the render path produces non-empty pixels.

## Current Status

Started implementation:

- Created the Swift package using Satin branch `feature/2.0-shader-source-transforms`.
- Added `SatinComputeRasteriser` library, demo executable, and tests.
- Added packed batch/file/pixel types and deterministic cube-grid fixture packing.
- Added `ComputeRasteriser`, `ComputeRasteriserPointCloud`, and compute processors for clear/depth/color/resolve.
- Added first Metal kernels for packed batch rendering, including reverse-Z depth with `atomic_max`.
- Added a minimal Satin demo renderer/view.
- Added layout and fixture-count tests.

Verified:

- `swift test` passes.
- `RasterBatch` stride is 64 bytes.
- `RasterFile` stride is 256 bytes.
- `RasterPixel` stride is 48 bytes because `uint3`/`SIMD3<UInt32>` has 16-byte alignment.

Next implementation risks:

- Metal shaders are resource-loaded and not compiled by `swift test`; the next step should launch or fixture-render to validate Satin shader reflection and runtime compilation.
- The first color pass uses simple atomics instead of NVIDIA subgroup partitioning, so correctness comes before performance.
- The reverse-Z color-pass tolerance may need visual tuning after the first rendered fixture.

## Metal Equivalent for CUDA Path and SIMD-Groups

The CUDA source path in `modules/compute_loop_las_cuda` should be treated as a separate Metal design, not a direct translation. Its core model differs from the current HQS accumulation path:

- One render kernel writes a single 64-bit winner per pixel.
- The packed winner is `(depth, pointID)`.
- A resolve kernel reads `pointID` from the winner buffer and fetches color from the source color buffer.
- It primarily uses 10-bit batch-local coordinates in the hot path, with older 20/30-bit branches mostly disabled/commented.

### Proposed Metal Pipelines

Keep two render modes behind the same public `ComputeRasteriser` object:

1. `highQualityAverage`
   - Current path.
   - Depth pass + color accumulation + resolve.
   - Better for averaging points near the visible surface.
2. `nearestPoint`
   - CUDA-inspired path.
   - Single render pass writes a packed winner.
   - Resolve pass fetches exact source color by point index.
   - Better starting point for CUDA parity and faster first visible result.

The baseline configuration enum is now in the public API:

```swift
public enum ComputeRasteriserMode {
    case highQualityAverage
    case nearestPoint
}
```

### Metal Winner Buffer Design

The first attempted design used one `device atomic_ulong*` or equivalent `RasterWinner` buffer:

```text
winner = (UInt64(depthReverseZ) << 32) | UInt64(pointIndex)
```

Reverse-Z means larger depth values are nearer, so winner selection is `atomic_max`, not CUDA's original `atomicMin`.

Runtime Metal compilation rejected `atomic_fetch_max` for `atomic_ulong` on the current toolchain. The implemented baseline therefore uses a portable two-pass design:

1. `NearestDepth` writes one reverse-Z `UInt32` depth per pixel with `atomic_max`.
2. `NearestIndex` reprojects the same points and writes the lowest matching `pointIndex` for pixels whose depth equals the winning depth.
3. `NearestResolve` reads `depth + pointIndex` and fetches `colors[pointIndex]`.

Clear values:

```text
depth = 0
pointIndex = 0xffffffff
```

Resolve:

```text
if winner == 0 -> background
pointIndex = winner & 0xffffffff
color = colors[pointIndex]
```

Risk: point indices are limited to 32 bits in this packing. That still covers up to 4.29B points, but the public data model should explicitly keep GPU point indices as `UInt32` unless we design a wider indirection table.

Fallback if 64-bit atomics are weak or unavailable:

- Store depth and point index in separate `atomic_uint` buffers.
- Use a compare-and-swap loop on depth, then update point index only when the depth update wins.
- This is less clean because depth/index coherence is harder under races; use only as fallback.

### SIMD-Group Strategy

The OpenGL HQS path uses NVIDIA subgroup partitioned reductions to combine points that hit the same pixel before doing global atomics. Metal does not provide a direct equivalent of `subgroupPartitionNV(pixelID)` for arbitrary dynamic keys. The Metal design should use staged reductions:

1. In-lane projection:
   - Each thread projects multiple points from a batch.
   - Compute `pixelID`, `depth`, `pointIndex`, and packed `winner`.
2. SIMD-group duplicate suppression:
   - Within a SIMD group, detect lanes targeting the same `pixelID` when possible.
   - Elect one lane to write the best winner among the matching lanes.
   - This reduces global atomics for coherent point order without requiring full partitioned subgroup support.
3. Threadgroup tile-local aggregation:
   - For stronger reduction, bin only points that fall inside a small screen tile owned by the threadgroup.
   - Accumulate into threadgroup memory, then flush one atomic per touched tile pixel.
   - This is the better Apple GPU equivalent for collision-heavy scenes.

Implementation order:

1. `NearestDepthProcessor` - baseline complete
   - Add `Pipelines/NearestDepth/Shaders.metal`.
   - One threadgroup per batch.
   - Decode packed positions and write `atomic_max` winner.
   - No SIMD-group optimization at first.
2. `NearestResolveProcessor` - baseline complete
   - Add `Pipelines/NearestResolve/Shaders.metal`.
   - Resolve winner buffer to output texture using `colors[pointIndex]`.
3. SIMD-group duplicate suppression experiment
   - Add a compile-time shader define, e.g. `CR_ENABLE_SIMD_DEDUP`.
   - Use Metal SIMD-group intrinsics where available.
   - Keep a plain atomic path for compatibility and verification.
4. Tile-local path experiment
   - Add a separate kernel, not a branch inside the baseline kernel.
   - Use fixed tile dimensions such as `16x16`.
   - Only enable after nearest-point correctness tests exist.

### Data Layout Changes

Add:

```swift
public struct RasterWinner {
    public var value: UInt64
}
```

or use raw `UInt64` buffers directly.

Add frame buffers:

- current HQS mode: `RasterPixel`
- nearest mode: `UInt64` winner buffer

Both modes can reuse:

- `RasterBatch`
- `RasterFile`
- `xyzLow/Med/High`
- `colors`

### Validation

Before optimizing with SIMD-groups:

1. Add CPU reference for nearest-point mode using the same reverse-Z projection rules. - complete
2. Render a deterministic fixture through nearest mode. - manual runtime smoke test complete; GPU readback test still pending
3. Read back a small output texture or winner buffer.
4. Compare selected pixels against CPU reference.
5. Only then add SIMD-group and tile-local variants.

Performance counters to record:

- points per second
- global atomic count estimate
- frame time for current HQS mode
- frame time for nearest mode
- frame time for SIMD dedup mode
- frame time for tile-local mode

### Non-Goals for Metal CUDA-Equivalent Pass

- CUDA/OpenGL interop parity.
- CUDA's exact `floatBitsToInt(pos.w)` depth ordering.
- Preserving the CUDA path's commented-out 20/30-bit branches until the 10-bit hot path is correct.
- Requiring Apple GPUs to emulate NVIDIA partitioned subgroup extensions exactly.
