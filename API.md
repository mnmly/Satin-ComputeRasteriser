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
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>
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
}
```

Defaults:

- `mode = .highQualityAverage`
- `depthTolerance = 0.01`
- `backgroundColor = [0, 0, 0, 0]`
- `enableFrustumCulling = true`
- `colorizeChunks = false`
- `colorizeOverdraw = false`

Modes:

- `.highQualityAverage`: depth pass, color accumulation, then resolve.
- `.nearestPoint`: CUDA-inspired path using a reverse-Z `UInt32` depth buffer, a second point-index pass, and `atomic_max` on depth.

The first nearest-point implementation resolves one visible point cloud per frame because the index buffer stores a local 32-bit point index, not a cloud id.

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
