import Metal
import Satin

open class NearestIndexProcessor: DepthPassProcessor {
    public var depthBuffer: MTLBuffer? {
        get { pixelBuffer }
        set { pixelBuffer = newValue }
    }

    public var indexBuffer: MTLBuffer? {
        didSet { set(indexBuffer, index: .Custom6) }
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: screenSize.x > 0
                && screenSize.y > 0
                && batchesBuffer != nil
                && depthBuffer != nil
                && indexBuffer != nil
                && visibleBatchesBuffer != nil
                && indirectArgsBuffer != nil
        )
    }
}
