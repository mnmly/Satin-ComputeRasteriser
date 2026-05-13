# Satin Compute Rasteriser

Satin Compute Rasteriser is a Swift package that ports the batch-oriented point-cloud compute rasterization path from Markus Schuetz's `compute_rasterizer` to Satin and Metal.

The current implementation focuses on the latest practical LAS/HQS path from the upstream repository, not every historical OpenGL/CUDA/VR/LOD variation. The default mode renders packed point clouds through a compute pipeline:

1. Clear per-pixel accumulation storage.
2. Run a batch-based depth pass.
3. Run a color accumulation pass.
4. Resolve accumulated color into a texture.
5. Composite the texture through Satin.

Satin uses reverse Z, so this port uses reverse-Z-compatible depth storage and `atomic_max` rather than copying the upstream OpenGL `atomicMin(floatBitsToInt(pos.w))` path literally.

There is also a CUDA-inspired `.nearestPoint` mode that writes reverse-Z per-pixel depth, resolves the matching point index in a second pass, and fetches color directly from the source point index. That path is separate from the HQS averaging path and is the baseline for future SIMD-group/tile-local optimization work.

## Status

Implemented:

- Swift package library target: `SatinComputeRasteriser`
- Packed point-cloud data model
- Deterministic fixture generation
- PLY point-cloud loading for `x/y/z` plus optional `red/green/blue`
- Clear, depth, color, and resolve Metal compute passes
- CUDA-inspired nearest-point Metal mode with portable 32-bit atomics
- Distance-based point sizing for both HQS and nearest modes
- Satin integration through `Object`, `ComputeProcessor`, and `PostProcessor`
- macOS example app in `Examples/SatinComputeRasteriserApp`

Not implemented yet:

- LAS/LAZ loader parity with upstream `LasLoaderSparse`
- Full CUDA path parity
- Potree node LOD path
- VR paths
- Bounding-box debug rendering
- NVIDIA subgroup partitioning optimization

## Requirements

- macOS 15 or newer
- Xcode with Metal tooling
- Swift 6 package toolchain

The package depends on Satin:

```swift
.package(url: "https://github.com/mnmly/Satin", branch: "feature/2.0-shader-source-transforms")
```

## Build and Test

Run the package tests:

```sh
swift test
```

Build the example app:

```sh
xcodebuild \
  -project Examples/SatinComputeRasteriserApp/SatinComputeRasteriserApp.xcodeproj \
  -scheme SatinComputeRasteriserApp \
  -destination platform=macOS \
  -derivedDataPath Examples/SatinComputeRasteriserApp/.build/xcode \
  CODE_SIGNING_ALLOWED=NO \
  build
```

`CODE_SIGNING_ALLOWED=NO` is useful for command-line verification. A normal Xcode GUI build can use the signing settings in the project.

## Example App

The example app lives outside the package manifest:

```text
Examples/SatinComputeRasteriserApp/SatinComputeRasteriserApp.xcodeproj
```

Run without arguments to render the built-in cube-grid fixture.

Run with a PLY file:

```sh
Examples/SatinComputeRasteriserApp/.build/xcode/Build/Products/Debug/SatinComputeRasteriserApp.app/Contents/MacOS/SatinComputeRasteriserApp \
  --ply /path/to/point-cloud.ply
```

Start in nearest-point mode:

```sh
Examples/SatinComputeRasteriserApp/.build/xcode/Build/Products/Debug/SatinComputeRasteriserApp.app/Contents/MacOS/SatinComputeRasteriserApp \
  --mode nearest
```

The app also has an `Open PLY` button using SwiftUI `fileImporter`.

Supported PLY input:

- `format ascii 1.0`
- `format binary_little_endian 1.0`
- vertex properties `x`, `y`, `z`
- optional color properties `red`, `green`, `blue`
- color aliases `r`, `g`, `b`, `diffuse_red`, `diffuse_green`, `diffuse_blue`

Unsupported for now:

- binary big-endian PLY
- list properties in the vertex element
- faces/meshes
- Gaussian splat-specific PLY attributes

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream license notices and research citations.

## API

See [API.md](API.md) for integration details.
