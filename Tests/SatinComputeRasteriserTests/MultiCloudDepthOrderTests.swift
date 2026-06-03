#if canImport(Metal)
import Foundation
import Metal
import Satin
import simd
import Testing
@testable import SatinComputeRasteriser

// Regression guard for cross-cloud depth reconciliation.
//
// `encodeHighQualityAverage` runs every cloud's depth pass before any cloud's
// color pass. The color pass commits a fragment's colour atomically and
// irreversibly when its depth is within tolerance of the pixel's winning depth.
// If depth+color were interleaved per cloud, a farther cloud iterated earlier
// would commit its colour before a nearer cloud lowered the pixel depth — and
// that colour could not be retracted. Occlusion would then depend on the order
// clouds were added, not on depth.
//
// These tests place two solid-colour cubes overlapping in screen space but
// separated in depth by far more than `depthTolerance` (hard occlusion), render
// them in BOTH insertion orders, and assert:
//   1. the overlap pixel is byte-identical between the two orders, and
//   2. the near (red) cloud wins — never the far (blue) one, never a blend.
// Pre-fix, the [far, near] order averaged to purple and both assertions failed.

private func makeContext() throws -> Context {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    return Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
}

/// A solid-colour cube lattice centred at the origin, spanning [-0.5, 0.5]^3.
private func solidCube(color: SIMD4<Float>, pointsPerAxis n: Int = 16) -> PackedPointCloud {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    positions.reserveCapacity(n * n * n)
    colors.reserveCapacity(n * n * n)
    for z in 0 ..< n {
        for y in 0 ..< n {
            for x in 0 ..< n {
                let fx = Float(x) / Float(n - 1)
                let fy = Float(y) / Float(n - 1)
                let fz = Float(z) / Float(n - 1)
                positions.append(SIMD3<Float>(fx - 0.5, fy - 0.5, fz - 0.5))
                colors.append(color)
            }
        }
    }
    return PackedPointCloudFixtures.pack(positions: positions, colors: colors)
}

/// Config that draws every resident point with a fat splat and crisp occlusion,
/// so the overlap pixel is deterministically covered by both cubes.
private func hardOcclusionConfig() -> ComputeRasteriserConfiguration {
    ComputeRasteriserConfiguration(
        mode: .highQualityAverage,
        depthTolerance: 0.002,                 // tight: 2-unit depth gap is hard occlusion
        backgroundColor: SIMD4<Float>(0, 0, 0, 1),
        enableFrustumCulling: false,
        enableCLOD: false,                     // draw ALL points, no LOD dropouts
        enableLODDither: false,
        holeFillIterations: 0,
        pointSizeMode: .screenSpace,
        minimumPointSize: 8,                   // fat splats → guaranteed centre coverage
        maximumPointSize: 64,
        pointSizeScale: 32
    )
}

/// Render two clouds (added in the given order) and read back the centre texel.
private func renderCentreTexel(
    context: Context,
    clouds: [ComputeRasteriserPointCloud],
    size: Int
) throws -> SIMD4<Float> {
    let device = context.device
    let queue = try #require(device.makeCommandQueue())

    let ras = ComputeRasteriser(context: context, label: "DepthOrderTest")
    ras.configuration = hardOcclusionConfig()
    for cloud in clouds { _ = ras.addPointCloud(cloud) }

    let viewport = SIMD4<Float>(0, 0, Float(size), Float(size))
    ras.resize(size: (Float(size), Float(size)))

    let camera = PerspectiveCamera(
        context: context, position: SIMD3<Float>(0, 0, 5), near: 0.1, far: 100, fov: 45
    )
    camera.lookAt(target: .zero)

    // A few frames to absorb any first-frame pipeline-compile latency.
    for _ in 0 ..< 3 {
        guard let commandBuffer = queue.makeCommandBuffer() else { continue }
        ras.update(renderContext: context, camera: camera, viewport: viewport, index: 0)
        ras.encode(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    let output = try #require(ras.outputTexture, "rasteriser produced no output texture")

    // Blit the private output texture into a shared buffer and read the centre.
    let bytesPerRow = size * 4
    let readback = try #require(
        device.makeBuffer(length: bytesPerRow * size, options: .storageModeShared)
    )
    let blitCB = try #require(queue.makeCommandBuffer())
    let blit = try #require(blitCB.makeBlitCommandEncoder())
    blit.copy(
        from: output,
        sourceSlice: 0, sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: size, height: size, depth: 1),
        to: readback,
        destinationOffset: 0,
        destinationBytesPerRow: bytesPerRow,
        destinationBytesPerImage: bytesPerRow * size
    )
    blit.endEncoding()
    blitCB.commit()
    blitCB.waitUntilCompleted()

    let px = readback.contents().bindMemory(to: UInt8.self, capacity: bytesPerRow * size)
    let cx = size / 2, cy = size / 2
    let o = cy * bytesPerRow + cx * 4
    return SIMD4<Float>(
        Float(px[o + 0]) / 255, Float(px[o + 1]) / 255,
        Float(px[o + 2]) / 255, Float(px[o + 3]) / 255
    )
}

@Test func overlapOcclusionIsIndependentOfCloudOrder() throws {
    let context = try makeContext()
    let size = 64

    // Near cube = red, sitting toward the camera (+Z); far cube = blue, behind
    // it. They overlap at the centre pixel; depth gap (2 units) ≫ tolerance.
    func near() -> ComputeRasteriserPointCloud {
        let c = ComputeRasteriserPointCloud(context: context, packed: solidCube(color: SIMD4(1, 0, 0, 1)), label: "near-red")
        c.position = SIMD3<Float>(0, 0, 1)
        return c
    }
    func far() -> ComputeRasteriserPointCloud {
        let c = ComputeRasteriserPointCloud(context: context, packed: solidCube(color: SIMD4(0, 0, 1, 1)), label: "far-blue")
        c.position = SIMD3<Float>(0, 0, -1)
        return c
    }

    // Fresh clouds per order so neither run mutates the other's GPU state.
    let texelNearFirst = try renderCentreTexel(context: context, clouds: [near(), far()], size: size)
    let texelFarFirst = try renderCentreTexel(context: context, clouds: [far(), near()], size: size)

    // (1) Occlusion must not depend on insertion order: byte-identical (±1/255).
    let delta = abs(texelNearFirst - texelFarFirst)
    #expect(simd_max(simd_max(delta.x, delta.y), simd_max(delta.z, delta.w)) <= 1.0 / 255 + 1e-4,
            "overlap texel differs by load order: nearFirst=\(texelNearFirst) farFirst=\(texelFarFirst)")

    // (2) The near (red) cloud must win in BOTH orders — never blue, never a blend.
    for (label, texel) in [("nearFirst", texelNearFirst), ("farFirst", texelFarFirst)] {
        #expect(texel.x > 0.6, "\(label): red channel too low — near cloud not winning: \(texel)")
        #expect(texel.z < 0.2, "\(label): blue channel too high — far cloud bleeding through: \(texel)")
    }
}

@Test func singleCloudRendersItsColor() throws {
    // Sanity: the trivial one-cloud path is unaffected by the two-phase split.
    let context = try makeContext()
    let size = 64
    let cloud = ComputeRasteriserPointCloud(context: context, packed: solidCube(color: SIMD4(1, 0, 0, 1)), label: "solo-red")
    let texel = try renderCentreTexel(context: context, clouds: [cloud], size: size)
    #expect(texel.x > 0.6 && texel.z < 0.2, "single red cloud did not render red: \(texel)")
}
#endif
