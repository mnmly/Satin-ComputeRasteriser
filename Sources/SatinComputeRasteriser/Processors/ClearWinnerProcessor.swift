import Metal
import Satin

open class ClearWinnerProcessor: BaseComputeRasteriserProcessor {
    public var pixelCount: Int = 0 {
        didSet { set("pixelCount", pixelCount) }
    }

    public var depthBuffer: MTLBuffer? {
        didSet { set(depthBuffer, index: .Custom0) }
    }

    public var indexBuffer: MTLBuffer? {
        didSet { set(indexBuffer, index: .Custom1) }
    }

    open override func setup() {
        super.setup()
        set("pixelCount", pixelCount)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: pixelCount > 0 && depthBuffer != nil && indexBuffer != nil)
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreads(
            MTLSize(width: pixelCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (pixelCount + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }
}
