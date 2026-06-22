#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// GPU integration coverage for `ComputeRasteriser.pickPointIndex(atNDC:in:camera:)`.
//
// The pick runs the same nearest-point pass the CPU oracle
// (`NearestPointCPUReference`) models, then reads back the winning global packed
// `pointIndex` for one pixel. These tests assert:
//   1. picking at a point's own projected location returns *that* point — and,
//      mapped through `PackedPointCloud.sourceIndices`, the original input index
//      (so the pick + the pack permutation round-trip end to end);
//   2. the GPU pick agrees with the CPU oracle at the picked pixel;
//   3. picking empty space returns nil (the `UInt32.max` sentinel).

private func makeContext() throws -> Context {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    return Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
}

/// Frustum-cull off, no LOD dropouts/dither, tight screen-space splats — so each
/// point lands on its own pixel and the GPU pass matches the radius-0 CPU oracle.
private func pickConfig() -> ComputeRasteriserConfiguration {
    ComputeRasteriserConfiguration(
        mode: .highQualityAverage,
        backgroundColor: SIMD4<Float>(0, 0, 0, 1),
        enableFrustumCulling: false,
        enableCLOD: false,
        enableLODDither: false,
        holeFillIterations: 0,
        pointSizeMode: .screenSpace,
        minimumPointSize: 1,
        maximumPointSize: 1,
        pointSizeScale: 1
    )
}

/// NDC (x, y, y-up) of `point` under `camera` — the convention `pickPointIndex`
/// and `Ray(camera:coordinate:)` expect.
private func ndc(of point: SIMD3<Float>, camera: Camera) -> SIMD2<Float> {
    let clip = camera.projectionMatrix * camera.viewMatrix * SIMD4<Float>(point, 1)
    return SIMD2<Float>(clip.x / clip.w, clip.y / clip.w)
}

@Test func pickReturnsTheProjectedPointAndRoundTripsThroughSourceIndices() throws {
    let context = try makeContext()
    let size = 64

    // Three points spread in X at equal depth — well separated at 64px, so each
    // owns a distinct pixel. Morton sort reorders them, which is exactly what
    // `sourceIndices` has to undo.
    let positions: [SIMD3<Float>] = [
        SIMD3<Float>(-0.8, 0, 0),
        SIMD3<Float>( 0.0, 0, 0),
        SIMD3<Float>( 0.8, 0, 0),
    ]
    let colors = [
        SIMD4<Float>(1, 0, 0, 1),
        SIMD4<Float>(0, 1, 0, 1),
        SIMD4<Float>(0, 0, 1, 1),
    ]
    let packed = PackedPointCloudFixtures.pack(positions: positions, colors: colors)
    #expect(packed.sourceIndices.count == positions.count, "pack must expose the permutation")

    let cloud = ComputeRasteriserPointCloud(context: context, packed: packed, label: "pick-test")
    let ras = ComputeRasteriser(context: context, label: "PickTest")
    ras.configuration = pickConfig()
    _ = ras.addPointCloud(cloud)

    let viewport = SIMD4<Float>(0, 0, Float(size), Float(size))
    ras.resize(size: (Float(size), Float(size)))

    let camera = PerspectiveCamera(context: context, position: SIMD3<Float>(0, 0, 5), near: 0.1, far: 100, fov: 45)
    camera.lookAt(target: .zero)

    // Warm up: a few frames make the cloud resident and absorb pipeline-compile
    // latency before the pick.
    let queue = try #require(context.device.makeCommandQueue())
    for _ in 0 ..< 3 {
        guard let cb = queue.makeCommandBuffer() else { continue }
        ras.update(renderContext: context, camera: camera, viewport: viewport, index: 0)
        ras.encode(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    let oracle = NearestPointCPUReference.render(
        packed: packed, width: size, height: size,
        viewMatrix: camera.viewMatrix, projectionMatrix: camera.projectionMatrix
    )

    for (originalIndex, position) in positions.enumerated() {
        let coordinate = ndc(of: position, camera: camera)
        let picked = try #require(
            ras.pickPointIndex(atNDC: coordinate, in: cloud, camera: camera),
            "pick at point \(originalIndex) returned nil"
        )

        // (1) packed index → original input index via the exposed permutation.
        #expect(Int(picked) < packed.sourceIndices.count)
        #expect(packed.sourceIndices[Int(picked)] == UInt32(originalIndex),
                "point \(originalIndex): picked packed index \(picked) maps to source \(packed.sourceIndices[Int(picked)])")

        // (2) GPU pick agrees with the CPU oracle at the same pixel.
        let px = min(max(Int((coordinate.x * 0.5 + 0.5) * Float(size)), 0), size - 1)
        let pyFromBottom = Int((coordinate.y * 0.5 + 0.5) * Float(size))
        let py = min(max(size - 1 - pyFromBottom, 0), size - 1)
        #expect(oracle.indices[py * size + px] == picked,
                "point \(originalIndex): GPU pick \(picked) disagrees with CPU oracle \(oracle.indices[py * size + px])")
    }
}

@Test func pickEmptySpaceReturnsNil() throws {
    let context = try makeContext()
    let size = 64

    let packed = PackedPointCloudFixtures.pack(
        positions: [SIMD3<Float>(0, 0, 0)],
        colors: [SIMD4<Float>(1, 1, 1, 1)]
    )
    let cloud = ComputeRasteriserPointCloud(context: context, packed: packed, label: "pick-empty")
    let ras = ComputeRasteriser(context: context, label: "PickEmptyTest")
    ras.configuration = pickConfig()
    _ = ras.addPointCloud(cloud)

    let viewport = SIMD4<Float>(0, 0, Float(size), Float(size))
    ras.resize(size: (Float(size), Float(size)))
    let camera = PerspectiveCamera(context: context, position: SIMD3<Float>(0, 0, 5), near: 0.1, far: 100, fov: 45)
    camera.lookAt(target: .zero)

    let queue = try #require(context.device.makeCommandQueue())
    for _ in 0 ..< 3 {
        guard let cb = queue.makeCommandBuffer() else { continue }
        ras.update(renderContext: context, camera: camera, viewport: viewport, index: 0)
        ras.encode(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    // A corner, far from the single centered point → no resident point.
    #expect(ras.pickPointIndex(atNDC: SIMD2<Float>(0.95, 0.95), in: cloud, camera: camera) == nil)
}
#endif
