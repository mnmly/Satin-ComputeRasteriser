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

    open override func setup() {
        super.setup()
        set("depthTolerance", depthTolerance)
        set("colorizeChunks", colorizeChunks ? 1 : 0)
        set("colorizeOverdraw", colorizeOverdraw ? 1 : 0)
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
