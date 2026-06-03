import Metal
import Satin
import simd

open class ResolveProcessor: BaseComputeRasteriserProcessor {
    public var width: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }

    public var height: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }

    public var backgroundColor: SIMD4<Float> = .zero {
        didSet { set("backgroundColor", backgroundColor) }
    }

    public var pixelBuffer: MTLBuffer? { didSet { set(pixelBuffer, index: .Custom0) } }
    public var outputTexture: MTLTexture? { didSet { set(outputTexture, index: .Custom0) } }
    /// Per-pixel reversed-Z NDC depth (R32Float; 0 = no cloud). Written
    /// alongside color so the composite can populate the scene depth buffer.
    public var depthTexture: MTLTexture? { didSet { set(depthTexture, index: .Custom1) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(width), Int32(height)))
        set("backgroundColor", backgroundColor)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: width > 0 && height > 0 && pixelBuffer != nil && outputTexture != nil && depthTexture != nil)
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let tw = max(pipeline.threadExecutionWidth, 1)
        let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        computeEncoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(tw, width), height: min(th, height), depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let tw = 16
        let th = 16
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (width + tw - 1) / tw, height: (height + th - 1) / th, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1)
        )
    }
}

