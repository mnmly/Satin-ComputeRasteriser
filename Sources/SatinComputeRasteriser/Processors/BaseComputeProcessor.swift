import Metal
import Satin

open class BaseComputeRasteriserProcessor: ComputeProcessor {
    /// Hook for subclasses to bind resources the base bind path can't express
    /// (e.g. an `MTLBuffer` with a non-zero offset that has to be re-applied
    /// after the standard `computeBuffers` binding, which always uses offset 0).
    open func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {}

    func encodeIfReady(_ commandBuffer: MTLCommandBuffer, isReady: Bool, configure: (MTLComputeCommandEncoder, MTLComputePipelineState) -> Void = { _, _ in }) {
        guard isReady,
              let pipeline = updatePipeline,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        update()
        computeEncoder.label = label
        computeEncoder.setComputePipelineState(pipeline)
        bindUniforms(computeEncoder)
        bindAllBuffers(computeEncoder)
        bindAllTextures(computeEncoder)
        applyAdditionalBindings(computeEncoder)
        configure(computeEncoder, pipeline)
        preCompute?(computeEncoder, 0)
        dispatchThreads(computeEncoder: computeEncoder, pipeline: pipeline, iteration: 0)
        computeEncoder.endEncoding()
    }

    private func bindAllBuffers(_ computeEncoder: MTLComputeCommandEncoder) {
        guard let shader else { return }
        for index in shader.bufferBindingIsUsed {
            if let uniformBuffer = computeUniformBuffers[index] {
                computeEncoder.setBuffer(uniformBuffer.buffer, offset: uniformBuffer.offset, index: index.rawValue)
            } else if let structBuffer = computeStructBuffers[index] {
                computeEncoder.setBuffer(structBuffer.buffer, offset: structBuffer.offset, index: index.rawValue)
            } else if let buffer = computeBuffers[index] {
                computeEncoder.setBuffer(buffer, offset: 0, index: index.rawValue)
            }
        }
    }

    private func bindAllTextures(_ computeEncoder: MTLComputeCommandEncoder) {
        guard let shader else { return }
        for index in shader.textureBindingIsUsed {
            if let texture = computeTextures[index] {
                computeEncoder.setTexture(texture, index: index.rawValue)
            }
        }
    }
}

