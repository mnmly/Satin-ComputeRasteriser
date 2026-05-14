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

For datasets larger than VRAM, the cloud is designed to be driven from
an external streaming source (e.g. SwiftPDAL's
`CopcStreamingPointCloudSource`). Non-resident slots short-circuit in
the cull kernel via ``RasterBatch/state`` — a `state == 0` batch is
skipped before frustum testing. The incremental upload API
(`addBatches`/`removeBatches`) is documented under
``ComputeRasteriserPointCloud`` once it lands; today's API is the
wholesale ``ComputeRasteriserPointCloud/replacePackedPointCloud(_:)``.

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
- ``PackedPointCloud``
- ``PackedPointCloudFixtures``
- ``PLYPointCloudLoader``

### GPU data layout (mirrored to Metal)

- ``RasterBatch``
- ``RasterFile``
- ``RasterPixel``
- ``VisibleBatch``
- ``CRDispatchArgs``
- ``ComputeRasteriserLayout``

### Reference

- <doc:CrossPackageLayout>
