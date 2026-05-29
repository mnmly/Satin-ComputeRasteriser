# Streaming COPC clouds

Drive a point cloud from an out-of-core source so the working set
stays under VRAM regardless of file size.

## Overview

The renderer's slot pool was designed to be paged — chunks load and
evict as the camera moves. The renderer doesn't own the loader; it
exposes ``ComputeRasteriserPointCloud/addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
and ``ComputeRasteriserPointCloud/removeBatches(slots:commit:)`` for an
external streaming source to call.

The reference implementation is
[SwiftPDAL](https://github.com/mnmly/SwiftPDAL)'s
`CopcStreamingPointCloudSource`, which reads
[COPC](https://copc.io) (Cloud Optimized Point Cloud) files via
copc-lib. The `SatinComputeRasteriserStreaming` library target ships a
small `StreamingAdapter` that wires the two together; the example app
under `Examples/SatinComputeRasteriserApp/` drives it.

## End-to-end shape

```swift
// 1. Open the source
let source = try await CopcStreamingPointCloudSource.open(url)
source.setBudget(1024 * 1024 * 1024) // 1 GB working set

// 2. Allocate a slot pool sized to your budget
let cap = ComputeRasteriserCapacity(
    maxResidentBatches: 8192,                        // tune by VRAM
    pointsPerBatch: source.info.pointsPerBatch       // matches source
)
let cloud = ComputeRasteriserPointCloud(context: ctx, capacity: cap)
rasteriser.addPointCloud(cloud)

// 3. Each frame: submit camera, drain delta, apply
var slotsByChunk: [ChunkID: [Int]] = [:]
func tick(camera: Camera, viewport: SIMD2<Float>) {
    source.submit(view: StreamingCameraView(
        position: camera.worldPosition,
        viewProjection: camera.projectionMatrix * camera.viewMatrix,
        pixelScale: viewport.y * 0.5
    ))
    guard let delta = source.pollLatest() else { return }

    // Evictions before adds so freed slots are available.
    var toFree: [Int] = []
    for id in delta.removed {
        if let slots = slotsByChunk.removeValue(forKey: id) {
            toFree.append(contentsOf: slots)
        }
    }
    cloud.removeBatches(slots: toFree)

    for chunk in delta.added {
        let slots = cloud.addBatches(
            positionsXYZLow: chunk.xyzLow,
            positionsXYZMed: chunk.xyzMed,
            positionsXYZHigh: chunk.xyzHigh,
            colors: chunk.colors,
            levels: chunk.levels,
            batches: chunk.batches.map(toRasterBatch)   // see below
        )
        slotsByChunk[chunk.id] = slots
    }
}
```

## Coordinate spaces — the precision trap

For georeferenced clouds (LAS/LAZ in projected CRS — UTM, State Plane,
etc.) absolute coordinates can be in the millions. FP32 has ~6 decimal
digits of precision, so at ~640,000 you have ~0.0625 unit precision per
axis — enough to make a cloud jitter violently or refuse to render.

**Don't** bake the source's `originShift` into ``RasterFile/world``:

```swift
// WRONG for georeferenced data: renders in absolute world coords
let cloud = ComputeRasteriserPointCloud(
    context: ctx,
    capacity: cap,
    originShift: SIMD3(source.info.originShift)   // ← FP32 death
)
```

**Do** keep the cloud in the source's pre-shifted space (chunks are
already small, FP32-safe values near origin) and frame the camera
against shifted bounds:

```swift
let cloud = ComputeRasteriserPointCloud(context: ctx, capacity: cap)
let shift = SIMD3<Float>(
    Float(source.info.originShift.x),
    Float(source.info.originShift.y),
    Float(source.info.originShift.z)
)
let shiftedMin = source.info.bounds.min - shift
let shiftedMax = source.info.bounds.max - shift
camera.lookAt(target: (shiftedMin + shiftedMax) * 0.5)
```

The user sees the cloud near origin instead of at its survey-grade
coordinates, which is what they want for visualization anyway.

## Why two batch types?

SwiftPDAL is renderer-agnostic, so it defines its own
`StreamingRasterBatch` value type with the **same byte layout** as this
package's ``RasterBatch``. Conversion is trivial:

```swift
private func toRasterBatch(_ s: StreamingRasterBatch) -> RasterBatch {
    var b = RasterBatch(
        min: SIMD3<Float>(s.minX, s.minY, s.minZ),
        max: SIMD3<Float>(s.maxX, s.maxY, s.maxZ),
        numPoints: s.numPoints,
        firstPoint: s.firstPoint,
        fileIndex: s.fileIndex
    )
    b.state = s.state
    return b
}
```

The layout match is pinned by `LayoutTests.rasterBatchFieldOffsetsMatchStreamingLayout`
in the renderer's test target — see <doc:CrossPackageLayout>.

## Sizing the slot pool

Per-slot byte cost on the GPU:

```
pointsPerBatch × 17 bytes
    = pointsPerBatch × (4 xyzLow + 4 xyzMed + 4 xyzHigh + 4 colors + 1 levels)
```

At the SwiftPDAL default `pointsPerBatch = 10240`, that's ~170 KB per
slot. For an 8 K slot pool: ~1.4 GB of point data.

Match `maxResidentBatches` to your VRAM budget. For dynamic budgets
(user-driven), recompute capacity from the budget at COPC-open time and
construct a fresh ``ComputeRasteriserPointCloud`` — the slot pool can't
grow at runtime today.

## Pitfalls

- **Don't** call ``ComputeRasteriserPointCloud/addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
  with more batches than ``ComputeRasteriserPointCloud/freeSlotCount``.
  The precondition will trap. Either evict first or skip and let the
  source retry next tick.
- **Always** preserve the `[ChunkID: [Int]]` map. Without it,
  ``ComputeRasteriserPointCloud/removeBatches(slots:commit:)`` won't know
  which slots a given evicted chunk owned.
- **Apply removes before adds** in each delta. New chunks may need
  slots that the removed chunks just freed.
- The cull kernel iterates over **every slot** every frame and short-
  circuits empty ones via ``RasterBatch/state``. Don't size the pool
  10× larger than you'll ever need; cull dispatches a threadgroup per
  slot regardless.
