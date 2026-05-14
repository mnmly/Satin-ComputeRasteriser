import Metal
import Satin

open class NearestDepthProcessor: DepthPassProcessor {
    public var winnerBuffer: MTLBuffer? {
        get { pixelBuffer }
        set { pixelBuffer = newValue }
    }

    /// NearestDepth's kernel does not declare a displacement buffer at Custom8,
    /// so we relax the parent's `displacementBuffer != nil` precondition.
    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: screenSize.x > 0
                && screenSize.y > 0
                && batchesBuffer != nil
                && pixelBuffer != nil
                && levelsBuffer != nil
                && visibleBatchesBuffer != nil
                && indirectArgsBuffer != nil
        )
    }
}
