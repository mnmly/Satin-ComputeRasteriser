import Metal
import Satin
import simd

open class ColorPassProcessor: DepthPassProcessor {
    public var depthTolerance: Float = 0.01 {
        didSet { set("depthTolerance", depthTolerance) }
    }

    public var colorizeChunks: Bool = false {
        didSet { set("colorizeChunks", colorizeChunks ? 1 : 0) }
    }

    public var colorizeOverdraw: Bool = false {
        didSet { set("colorizeOverdraw", colorizeOverdraw ? 1 : 0) }
    }

    public var colorsBuffer: MTLBuffer? { didSet { set(colorsBuffer, index: .Custom9) } }

    // `applyTint`, `tintAlphaIsCoverage`, and `tintBuffer` (Custom10) are inherited
    // from `DepthPassProcessor` so both passes share one tint binding + flags.

    // MARK: - Motion blur (color-pass-only coverage/OIT smear)

    public var motionBlur: Float = 0.0 {
        didSet { set("motionBlur", motionBlur) }
    }
    public var motionBlurSamples: Int = 8 {
        didSet { set("motionBlurSamples", motionBlurSamples) }
    }
    public var motionBlurMaxSpread: Float = 64.0 {
        didSet { set("motionBlurMaxSpread", motionBlurMaxSpread) }
    }

    /// Previous frame's displacement, bound at raw buffer index 12 (an unused
    /// compute slot — the `ComputeBufferIndex` enum stops at Custom10). Read by
    /// the color kernel to build per-point displacement velocity. Bound to a
    /// stand-in (the current displacement) when nil so Metal validation passes
    /// and a still cloud reads zero displacement velocity.
    public var prevDisplacementBuffer: MTLBuffer?
    static let prevDisplacementIndex = 12

    open override func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {
        super.applyAdditionalBindings(computeEncoder)
        let prev = prevDisplacementBuffer ?? displacementBuffer
        if let prev {
            computeEncoder.setBuffer(prev, offset: 0, index: Self.prevDisplacementIndex)
        }
    }

    open override func setup() {
        super.setup()
        set("depthTolerance", depthTolerance)
        set("colorizeChunks", colorizeChunks ? 1 : 0)
        set("colorizeOverdraw", colorizeOverdraw ? 1 : 0)
        set("motionBlur", motionBlur)
        set("motionBlurSamples", motionBlurSamples)
        set("motionBlurMaxSpread", motionBlurMaxSpread)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: screenSize.x > 0
                && screenSize.y > 0
                && batchesBuffer != nil
                && colorsBuffer != nil
                && pixelBuffer != nil
                && levelsBuffer != nil
                && visibleBatchesBuffer != nil
                && displacementBuffer != nil
                && tintBuffer != nil
                && indirectArgsBuffer != nil
        )
    }
}
