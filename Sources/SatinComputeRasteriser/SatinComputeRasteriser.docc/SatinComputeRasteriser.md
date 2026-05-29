# ``SatinComputeRasteriser``

A Metal compute-shader point-cloud rasterizer for Satin. Ported from
Markus Schuetz's `compute_rasterizer` and extended with techniques from
Magnopus's large-cloud rendering write-up.

## Overview

The renderer hands the GPU a packed point cloud and dispatches a chain of
compute kernels per frame: cull batches → dispatch indirect → fill a
pixel buffer (depth or atomic-average color) → resolve to a texture →
optional hole-fill. Output is composited as a fullscreen quad.

You drive it from a Satin scene tree:

```swift
let rasteriser = ComputeRasteriser(context: context)
let cloud = ComputeRasteriserPointCloud(context: context, packed: packed)
rasteriser.addPointCloud(cloud)
scene.add(rasteriser)
```

`ComputeRasteriser` is itself a Satin `Object` — add it as a child of any
scene and call `draw(renderPassDescriptor:commandBuffer:)` from your
render loop. Point clouds are children of the rasteriser; you can mix
multiple clouds and toggle them with `visible`.

### Streaming

For datasets larger than VRAM, the cloud is driven from an external
streaming source (e.g. SwiftPDAL's `CopcStreamingPointCloudSource`).
Construct the cloud with ``ComputeRasteriserPointCloud/init(context:capacity:files:originShift:label:)``
to allocate a fixed slot pool, then page chunks in/out via
``ComputeRasteriserPointCloud/addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
and ``ComputeRasteriserPointCloud/removeBatches(slots:commit:)`` as the camera
moves. Non-resident slots short-circuit in the cull kernel via
``RasterBatch/state`` — a `state == 0` batch is skipped before frustum
testing.

### Point packing

Positions are quantized to 30 bits per axis and split across three
`UInt32` buffers (`xyzLow`/`Med`/`High`, 10 bits each). The packing math
lives in `PackedPointCloudFixtures.pack(...)` and is intentionally
mirrored in SwiftPDAL's `ChunkPacker` so streamed chunks are upload-ready.

## Topics

### Driving the renderer

- ``ComputeRasteriser``
- ``ComputeRasteriserConfiguration``
- ``ComputeRasteriserMode``
- ``PointSizeMode``

### Point clouds

- ``ComputeRasteriserPointCloud``
- ``ComputeRasteriserCapacity``
- ``PackedPointCloud``
- ``PackedPointCloudFixtures``
- ``PLYPointCloudLoader``

### Per-point overrides

- ``DisplacementPass``
- ``TintPass``

### GPU data layout (mirrored to Metal)

- ``RasterBatch``
- ``RasterFile``
- ``RasterPixel``
- ``VisibleBatch``
- ``CRDispatchArgs``
- ``ComputeRasteriserLayout``

### Streaming companion

The `SatinComputeRasteriserStreaming` library product (separate target
in this package) provides `StreamingAdapter`, the SwiftPDAL glue. Link it
only if you want to drive a cloud from a SwiftPDAL streaming source.

### Guides

- <doc:StreamingCOPCClouds>
- <doc:ConvertingPLYToCOPC>

### Reference

- <doc:CrossPackageLayout>
