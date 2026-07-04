import simd

/// Projection-math helpers used by streaming sources to convert renderer
/// state into screen-space units.
public enum ComputeRasteriserProjection {
    /// Scale-invariant screen-space-pixel factor used by streaming residency
    /// scorers (e.g. SwiftPDAL's `StreamingCameraView.pixelScale`):
    ///
    /// ```
    /// pixelScale = viewportHeight / (2 · tan(fovY / 2))
    ///            = viewportHeight × 0.5 × projectionMatrix[1][1]
    /// ```
    ///
    /// `projectionMatrix[1][1]` equals `1 / tan(fovY / 2)` for any
    /// perspective matrix (Satin's `PerspectiveCamera.projectionMatrix`,
    /// custom-built rigs, stereo-skewed projections — all the same).
    /// Using the matrix form avoids depending on a particular camera type
    /// or having to track the FOV separately.
    ///
    /// **Why scale-invariant matters.** A 1 m scene viewed from 2 m and a
    /// 1000 m scene viewed from 2000 m give identical `extent / distance`,
    /// so the chunk's screen-space size — and therefore its desired LOD
    /// depth — is the same. SwiftPDAL's scorer relies on this; passing a
    /// non-FOV-aware constant breaks it (every camera gets the same
    /// "size-on-screen" verdict regardless of zoom).
    ///
    /// **Orthographic projections work too.** For an orthographic matrix
    /// `projectionMatrix[1][1] = 2 / orthoHeight`, so this same expression
    /// yields the absolute (depth-independent) pixels-per-world-unit scale
    /// rather than a FOV-derived one. SwiftPDAL's scorer consumes both forms
    /// ortho-aware — no distance division is applied for the orthographic case.
    ///
    /// - Parameters:
    ///   - viewportHeight: drawable height in pixels.
    ///   - projectionMatrix: any perspective or orthographic projection matrix
    ///     in column-major (Metal/simd) layout.
    @inlinable
    public static func screenSpacePixelScale(
        viewportHeight: Float,
        projectionMatrix: simd_float4x4
    ) -> Float {
        viewportHeight * 0.5 * projectionMatrix.columns.1.y
    }
}
