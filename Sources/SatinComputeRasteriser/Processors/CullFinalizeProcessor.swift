import Metal
import Satin

open class CullFinalizeProcessor: BaseComputeRasteriserProcessor {
    public var counterBuffer: MTLBuffer? { didSet { set(counterBuffer, index: .Custom0) } }
    public var indirectArgsBuffer: MTLBuffer? { didSet { set(indirectArgsBuffer, index: .Custom1) } }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: counterBuffer != nil && indirectArgsBuffer != nil
        )
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreads(
            MTLSize(width: 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }
}
