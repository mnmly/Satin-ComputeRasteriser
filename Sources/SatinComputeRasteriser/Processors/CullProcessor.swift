import Metal
import Satin
import simd

open class CullProcessor: BaseComputeRasteriserProcessor {
    public var batchCount: Int = 0 {
        didSet { set("batchCount", batchCount) }
    }

    public var screenSize: SIMD2<UInt32> = .zero {
        didSet { set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y))) }
    }

    public var viewMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("viewMatrix", viewMatrix) }
    }

    public var projectionMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("projectionMatrix", projectionMatrix) }
    }

    public var enableFrustumCulling: Bool = true {
        didSet { set("enableFrustumCulling", enableFrustumCulling ? 1 : 0) }
    }

    public var lodBias: Int = 0 {
        didSet { set("lodBias", lodBias) }
    }

    public var batchesBuffer: MTLBuffer? { didSet { set(batchesBuffer, index: .Custom0) } }
    public var filesBuffer: (any BindableBuffer)? { didSet { set(filesBuffer, index: .Custom1) } }
    public var visibleBuffer: MTLBuffer? { didSet { set(visibleBuffer, index: .Custom2) } }
    public var counterBuffer: MTLBuffer? { didSet { set(counterBuffer, index: .Custom3) } }

    open override func setup() {
        super.setup()
        set("batchCount", batchCount)
        set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y)))
        set("viewMatrix", viewMatrix)
        set("projectionMatrix", projectionMatrix)
        set("enableFrustumCulling", enableFrustumCulling ? 1 : 0)
        set("lodBias", lodBias)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: batchCount > 0
                && batchesBuffer != nil
                && filesBuffer != nil
                && visibleBuffer != nil
                && counterBuffer != nil
        )
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        computeEncoder.dispatchThreads(
            MTLSize(width: batchCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: (batchCount + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }
}
