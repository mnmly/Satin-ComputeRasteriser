import Metal
import Satin
import simd

open class HoleFillProcessor: BaseComputeRasteriserProcessor {
    public var width: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }

    public var height: Int = 0 {
        didSet { set("screenSize", simd_int2(Int32(width), Int32(height))) }
    }

    public var inputTexture: MTLTexture? { didSet { set(inputTexture, index: .Custom0) } }
    public var outputTexture: MTLTexture? { didSet { set(outputTexture, index: .Custom1) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(width), Int32(height)))
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: width > 0 && height > 0 && inputTexture != nil && outputTexture != nil)
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
