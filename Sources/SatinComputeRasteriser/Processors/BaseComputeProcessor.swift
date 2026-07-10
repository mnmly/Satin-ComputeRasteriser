import Metal
import Satin

open class BaseComputeRasteriserProcessor: ComputeProcessor {
    /// Hook for subclasses to bind resources the base bind path can't express
    /// (e.g. an `MTLBuffer` with a non-zero offset that has to be re-applied
    /// after the standard `computeBuffers` binding, which always uses offset 0).
    open func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {}

    /// Encode one dispatch into a caller-owned encoder: pipeline state, uniform
    /// slot, buffer/texture bindings, ``applyAdditionalBindings(_:)``,
    /// `configure`, `preCompute`, dispatch. Does **not** create/end the encoder
    /// and does **not** call `update()` — the caller advances the uniform ring
    /// exactly once per frame before its dispatch loop, so every dispatch
    /// sharing the encoder reads the same uniform slot. Per-dispatch values must
    /// therefore be raw buffer bindings or `setBytes`, never uniform parameters.
    func encode(
        into computeEncoder: MTLComputeCommandEncoder,
        isReady: Bool = true,
        configure: (MTLComputeCommandEncoder, MTLComputePipelineState) -> Void = { _, _ in }
    ) {
        guard isReady, let pipeline = updatePipeline else { return }
        computeEncoder.setComputePipelineState(pipeline)
        bindUniforms(computeEncoder)
        bindAllBuffers(computeEncoder)
        bindAllTextures(computeEncoder)
        applyAdditionalBindings(computeEncoder)
        configure(computeEncoder, pipeline)
        preCompute?(computeEncoder, 0)
        dispatchThreads(computeEncoder: computeEncoder, pipeline: pipeline, iteration: 0)
    }

    /// Single-dispatch convenience: makes its own compute encoder, advances the
    /// uniform ring via `update()`, encodes through ``encode(into:isReady:configure:)``,
    /// and ends the encoder.
    func encodeIfReady(_ commandBuffer: MTLCommandBuffer, isReady: Bool, configure: (MTLComputeCommandEncoder, MTLComputePipelineState) -> Void = { _, _ in }) {
        guard isReady,
              updatePipeline != nil,
              let computeEncoder = commandBuffer.makeComputeCommandEncoder()
        else { return }

        update()
        computeEncoder.label = label
        encode(into: computeEncoder, isReady: isReady, configure: configure)
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

