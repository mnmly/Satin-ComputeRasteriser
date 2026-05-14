# API

This package exposes a small API around packed point-cloud rendering. The public surface is intentionally narrow while the port is still early.

## Main Types

### `ComputeRasteriser`

`ComputeRasteriser` is a Satin `Object` that owns the compute pipeline and output texture.

```swift
let rasteriser = ComputeRasteriser(context: context)
scene.add(rasteriser)
```

Use it inside a Satin renderer:

```swift
renderer.draw(
    renderPassDescriptor: renderPassDescriptor,
    commandBuffer: commandBuffer,
    scene: scene,
    camera: camera
)

rasteriser.draw(
    renderPassDescriptor: renderPassDescriptor,
    commandBuffer: commandBuffer
)
```

Call `resize` from your view renderer resize path:

```swift
rasteriser.resize(size: size, scaleFactor: scaleFactor)
```

Key properties:

```swift
public var configuration: ComputeRasteriserConfiguration
public private(set) var outputTexture: MTLTexture?
```

Point cloud management:

```swift
@discardableResult
public func addPointCloud(_ cloud: ComputeRasteriserPointCloud) -> ComputeRasteriserPointCloud

public func removePointCloud(_ cloud: ComputeRasteriserPointCloud)

public var pointClouds: [ComputeRasteriserPointCloud]
```

### `ComputeRasteriserPointCloud`

`ComputeRasteriserPointCloud` is a Satin `Object` that owns GPU buffers for one packed point cloud.

```swift
let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 28)
let cloud = ComputeRasteriserPointCloud(context: context, packed: packed)
rasteriser.addPointCloud(cloud)
```

Replace the data after loading:

```swift
let packed = try PLYPointCloudLoader.load(url: url)
cloud.replacePackedPointCloud(packed)
```

Because it is an `Object`, the point cloud can be transformed through Satin object transforms:

```swift
cloud.position = [0, 0, -2]
cloud.scale = [2, 2, 2]
```

### Per-point displacement (animated noise, deformers, jitter)

`ComputeRasteriserPointCloud.displacementBuffer` is an optional `MTLBuffer`
of `float3` deltas (stride 16, one entry per pack-order `pointIndex`).
When `ComputeRasteriserConfiguration.applyDisplacement == true`, the
**DepthPass** and **ColorPass** kernels add `displacements[pointIndex]` to
each decoded position *before* projection. Both passes see the same
delta, so depth and color land on the same pixel.

Only the `.highQualityAverage` mode honors displacement today; the
`.nearestPoint` path is unchanged.

**Recipe — fill displacement from a custom compute kernel each frame:**

```swift
// 1. Pack. PackedPointCloud.orderedPositions is the user's original
//    positions reordered into pack-order — the same index the rasteriser
//    uses, so a displacement kernel can read them 1:1.
let packed = PackedPointCloudFixtures.pack(positions: positions, colors: colors)
let cloud = ComputeRasteriserPointCloud(context: context, packed: packed)

let originalBuf = packed.orderedPositions.withUnsafeBytes { bytes in
    device.makeBuffer(
        bytes: bytes.baseAddress!,
        length: packed.orderedPositions.count * MemoryLayout<SIMD3<Float>>.stride,
        options: .storageModeShared
    )
}

// 2. Allocate the displacement buffer the rasteriser will read.
//    `.private` storage is fine — only the GPU touches it.
cloud.displacementBuffer = cloud.makeDisplacementBuffer(storage: .private)
rasteriser.addPointCloud(cloud)

// 3. Enable on the rasteriser.
rasteriser.configuration.applyDisplacement = true

// 4. Every frame, before super.draw(...), encode your kernel that reads
//    originalBuf and writes deltas into cloud.displacementBuffer. Metal
//    serializes compute → compute on the same command buffer, so the
//    rasteriser's depth/color passes see your writes.
```

Custom kernel signature:

```metal
kernel void computeDisplacement(
    device const float3 *originalPositions [[buffer(0)]],
    device       float3 *displacements     [[buffer(1)]],
    constant     NoiseParams &noise        [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= noise.count) return;
    const float3 p = originalPositions[id];
    displacements[id] = snoise3(p * noise.freq) * noise.amp;
}
```

Notes / gotchas:

- **Deltas, not absolute positions.** The shader does `point += delta`
  after `decodePoint`. Writing the displaced position into the buffer
  would double-up the original.
- **Pack-order indexing.** The Morton sort inside `pack(...)` reorders
  points. `packed.orderedPositions[i]` is the original position for
  pack-order index `i`, which is also the index the rasteriser shader
  uses. Indexing by the user's pre-pack array order will scramble the
  result.
- **Frustum culling sees undisplaced bounds.** Large displacements can
  push points outside their batch's AABB and trigger pops at the edge of
  the view. Disable `enableFrustumCulling` or pad your batch bounds if
  you go beyond ~one batch radius.
- **Metal validation.** `displacementBuffer` must be bound at the
  `Custom8` slot whenever the depth/color processors run; the
  rasteriser binds `xyzLowBuffer` as a harmless stand-in when the cloud
  has no real displacement buffer, so `applyDisplacement = false` works
  with any cloud.
- **Static displacement** (e.g. one-shot scatter) works the same way —
  populate `displacementBuffer` once at load time and leave
  `applyDisplacement = true`.

### `PackedPointCloud`

`PackedPointCloud` is CPU-side packed point data.

```swift
public struct PackedPointCloud {
    public var batches: [RasterBatch]
    public var files: [RasterFile]
    public var xyzLow: [UInt32]
    public var xyzMed: [UInt32]
    public var xyzHigh: [UInt32]
    public var colors: [UInt32]
    public var levels: [UInt8]
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>
    /// Original positions reordered into pack-order. Filled by
    /// `PackedPointCloudFixtures.pack(...)`. Empty when a loader didn't
    /// preserve them (e.g. paged PLY loaders). Useful for keying a
    /// per-point displacement buffer to the same indices the rasteriser
    /// reads.
    public var orderedPositions: [SIMD3<Float>]
}
```

Coordinates are encoded per batch into three 10-bit chunks:

- `xyzLow`: highest 10 bits per axis
- `xyzMed`: middle 10 bits per axis
- `xyzHigh`: lowest 10 bits per axis

Colors are packed as:

```swift
R | (G << 8) | (B << 16)
```

### `ComputeRasteriserConfiguration`

```swift
public struct ComputeRasteriserConfiguration {
    public var mode: ComputeRasteriserMode
    public var depthTolerance: Float
    public var backgroundColor: SIMD4<Float>
    public var enableFrustumCulling: Bool
    public var colorizeChunks: Bool
    public var colorizeOverdraw: Bool
    public var minimumPointSize: Float
    public var maximumPointSize: Float
    public var pointSizeScale: Float
    /// When true, the depth + color passes add a per-point `float3` from
    /// `cloud.displacementBuffer` (Custom8) after decoding the position.
    /// See "Per-point displacement" above.
    public var applyDisplacement: Bool
}
```

Defaults:

- `mode = .highQualityAverage`
- `depthTolerance = 0.01`
- `backgroundColor = [0, 0, 0, 0]`
- `enableFrustumCulling = true`
- `colorizeChunks = false`
- `colorizeOverdraw = false`
- `minimumPointSize = 1`
- `maximumPointSize = 1`
- `pointSizeScale = 1`

Modes:

- `.highQualityAverage`: depth pass, color accumulation, then resolve.
- `.nearestPoint`: CUDA-inspired path using a reverse-Z `UInt32` depth buffer, a second point-index pass, and `atomic_max` on depth.

The first nearest-point implementation resolves one visible point cloud per frame because the index buffer stores a local 32-bit point index, not a cloud id.

Point size:

```swift
rasteriser.configuration.minimumPointSize = 1
rasteriser.configuration.maximumPointSize = 5
rasteriser.configuration.pointSizeScale = 5
```

The shader computes `pointSize = clamp(pointSizeScale / cameraDistance, minimumPointSize, maximumPointSize)`. A size of `1` writes a single pixel, so the defaults preserve the original one-pixel rasterization.

## Loading PLY

Load a PLY file directly into a packed point cloud:

```swift
let packed = try PLYPointCloudLoader.load(url: url)
```

Or parse existing data:

```swift
let packed = try PLYPointCloudLoader.parse(data)
```

Supported PLY formats:

- ASCII
- binary little-endian

Required vertex properties:

- `x`
- `y`
- `z`

Optional vertex color properties:

- `red`, `green`, `blue`
- `r`, `g`, `b`
- `diffuse_red`, `diffuse_green`, `diffuse_blue`

If no color properties are present, points default to white.

## Fixture Data

Use fixture data for smoke tests and examples:

```swift
let packed = PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 24)
```

Pack arrays directly:

```swift
let packed = PackedPointCloudFixtures.pack(
    positions: positions,
    colors: colors
)
```

## Layout Constants

The tests assert shader-compatible layout:

```swift
ComputeRasteriserLayout.rasterBatchStride == 64
ComputeRasteriserLayout.rasterFileStride == 256
ComputeRasteriserLayout.rasterPixelStride == 48
```

`RasterPixel` is 48 bytes because Metal `uint3`/Swift `SIMD3<UInt32>` has 16-byte alignment.

## Runtime Notes

- The renderer writes to an offscreen `.rgba8Unorm` texture and composites it through Satin.
- The compute passes use reverse-Z depth.
- The first color pass uses simple atomics for portability on Apple GPUs. It does not yet implement the upstream NVIDIA subgroup partitioning optimization.
- Satin runtime-compiles the Metal source from the package resource bundle, so shader errors usually appear when the example app is launched, not during `swift test`.
