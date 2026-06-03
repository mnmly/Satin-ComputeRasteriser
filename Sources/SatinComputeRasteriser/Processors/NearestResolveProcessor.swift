import Metal
import Satin
import simd

open class NearestResolveProcessor: ResolveProcessor {
    public var depthBuffer: MTLBuffer? {
        get { pixelBuffer }
        set { pixelBuffer = newValue }
    }

    public var indexBuffer: MTLBuffer? { didSet { set(indexBuffer, index: .Custom1) } }
    public var colorsBuffer: MTLBuffer? { didSet { set(colorsBuffer, index: .Custom2) } }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: width > 0 && height > 0 && pixelBuffer != nil && indexBuffer != nil && colorsBuffer != nil && outputTexture != nil && depthTexture != nil)
    }
}
