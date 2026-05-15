import Foundation
import simd
import Testing
@testable import SatinComputeRasteriser

// SwiftPDAL's residency scorer expects a scale-invariant pixelScale
// (a 1 m scene at 2 m and a 1000 m scene at 2 km should produce the same
// "size on screen" verdict). The naive `viewportHeight * 0.5` we shipped
// initially is FOV-blind and breaks this — these tests pin the correct
// formula and the angular-size identity it implies.

private func perspectiveProj(fovYDegrees: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let f = 1.0 / tan(fovYDegrees * 0.5 * .pi / 180)
    var m = matrix_identity_float4x4
    m.columns.0 = SIMD4<Float>(f / aspect, 0, 0, 0)
    m.columns.1 = SIMD4<Float>(0, f, 0, 0)
    m.columns.2 = SIMD4<Float>(0, 0, far / (near - far), -1)
    m.columns.3 = SIMD4<Float>(0, 0, (near * far) / (near - far), 0)
    return m
}

@Test func pixelScaleMatchesAnalyticFOVFormula() {
    let fovYDeg: Float = 45
    let viewportH: Float = 1000
    let proj = perspectiveProj(fovYDegrees: fovYDeg, aspect: 16.0/9.0, near: 0.1, far: 1000)

    let pixelScale = ComputeRasteriserProjection.screenSpacePixelScale(
        viewportHeight: viewportH, projectionMatrix: proj
    )
    let expected = viewportH * 0.5 / tan(fovYDeg * 0.5 * .pi / 180)

    #expect(abs(pixelScale - expected) < 0.01,
            "got \(pixelScale), expected \(expected)")
}

@Test func pixelScaleIsScaleInvariantUnderAngularSizeIdentity() {
    // Same camera, two very different scene scales: extent/distance is
    // identical so screenPx must be identical.
    let proj = perspectiveProj(fovYDegrees: 45, aspect: 1, near: 0.1, far: 1e6)
    let pixelScale = ComputeRasteriserProjection.screenSpacePixelScale(
        viewportHeight: 1000, projectionMatrix: proj
    )

    let screenPx_1m  = (1.0 as Float)    * pixelScale / 2.0
    let screenPx_1km = (1000.0 as Float) * pixelScale / 2000.0

    #expect(abs(screenPx_1m - screenPx_1km) < 0.001,
            "1m@2m gave \(screenPx_1m), 1km@2km gave \(screenPx_1km)")
}

@Test func pixelScaleResponseToFOV() {
    // Tighter FOV → larger pixelScale (one degree of view fills more
    // pixels). At 1000 px viewport: 90° → 500, 45° → ~1207, 10° → ~5715.
    let proj90 = perspectiveProj(fovYDegrees: 90, aspect: 1, near: 0.1, far: 1000)
    let proj45 = perspectiveProj(fovYDegrees: 45, aspect: 1, near: 0.1, far: 1000)
    let proj10 = perspectiveProj(fovYDegrees: 10, aspect: 1, near: 0.1, far: 1000)
    let s90 = ComputeRasteriserProjection.screenSpacePixelScale(viewportHeight: 1000, projectionMatrix: proj90)
    let s45 = ComputeRasteriserProjection.screenSpacePixelScale(viewportHeight: 1000, projectionMatrix: proj45)
    let s10 = ComputeRasteriserProjection.screenSpacePixelScale(viewportHeight: 1000, projectionMatrix: proj10)

    #expect(abs(s90 - 500.0) < 1.0)
    #expect(abs(s45 - 1207.0) < 5.0)
    #expect(abs(s10 - 5715.0) < 30.0)
    #expect(s10 > s45 && s45 > s90)
}

@Test func screenPxForNormallyFramedSceneIsHundreds() {
    // Sanity check from the SwiftPDAL agent's note: for a "normally
    // framed" view (camera positioned so the scene fills ~70% of the
    // vertical FOV), screenPx for the full bounds extent should be in
    // the hundreds — not 10s, not 10000s. If this ever drifts,
    // SwiftPDAL's targetChunkScreenSize (default 256) is being scored
    // against wrong inputs and residency selection gets pathological.
    let span: Float = 50
    let fovYDeg: Float = 45
    let viewportH: Float = 1000
    let proj = perspectiveProj(fovYDegrees: fovYDeg, aspect: 16.0/9.0, near: 0.01, far: 1000)
    let cameraDistance = (span * 0.5) / tan(fovYDeg * 0.5 * .pi / 180) * 1.35   // matches frameCamera factor

    let pixelScale = ComputeRasteriserProjection.screenSpacePixelScale(
        viewportHeight: viewportH, projectionMatrix: proj
    )
    let screenPx = span * pixelScale / cameraDistance

    #expect(screenPx > 200 && screenPx < 1500,
            "screenPx \(screenPx) outside expected hundreds — pixelScale or framing math regressed")
}

@Test func screenPxIsScaleInvariantAcrossSceneSizes() {
    // The renderer's frameCamera positions the camera at
    // distance ≈ (span/2) / tan(fovY/2) * 1.35 — proportional to span.
    // So screenPx for the full bounds should be identical regardless of
    // scene scale. This is the property that lets a 1m vs 1km cloud
    // load with the same residency depth profile.
    let proj = perspectiveProj(fovYDegrees: 45, aspect: 1, near: 0.0001, far: 1e7)
    let pixelScale = ComputeRasteriserProjection.screenSpacePixelScale(
        viewportHeight: 1000, projectionMatrix: proj
    )
    func screenPxForFramedScene(span: Float) -> Float {
        let dist = (span * 0.5) / tan(45.0 * 0.5 * .pi / 180) * 1.35
        return span * pixelScale / dist
    }
    #expect(abs(screenPxForFramedScene(span: 1)    - screenPxForFramedScene(span: 1000)) < 0.001)
    #expect(abs(screenPxForFramedScene(span: 1000) - screenPxForFramedScene(span: 1_000_000)) < 0.001)
}
