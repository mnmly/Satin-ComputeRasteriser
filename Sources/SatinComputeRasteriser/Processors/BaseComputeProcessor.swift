import Metal
import Satin

open class BaseComputeRasteriserProcessor: ComputeProcessor {
    func encodeIfReady(_ commandBuffer: MTLCommandBuffer, isReady: Bool, configure: (MTLComputeCommandEncoder, MTLComputePipelineState) -> Void = { _, _ in }) {
        guard isReady,
              let pipeline = updatePipeline,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        update()
        computeEncoder.label = label
        computeEncoder.setComputePipelineState(pipeline)
        bindUniforms(computeEncoder)
        bindBuffers(computeEncoder)
        bindTextures(computeEncoder)
        configure(computeEncoder, pipeline)
        preCompute?(computeEncoder, 0)
        dispatch(computeEncoder: computeEncoder, pipeline: pipeline, iteration: 0)
        computeEncoder.endEncoding()
    }
}

