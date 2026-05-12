import Metal
import Satin
import simd

open class DepthPassProcessor: BaseComputeRasteriserProcessor {
    public var batchCount: Int = 0

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

    public var batchesBuffer: MTLBuffer? { didSet { set(batchesBuffer, index: .Custom0) } }
    public var xyzLowBuffer: MTLBuffer? { didSet { set(xyzLowBuffer, index: .Custom1) } }
    public var xyzMedBuffer: MTLBuffer? { didSet { set(xyzMedBuffer, index: .Custom2) } }
    public var xyzHighBuffer: MTLBuffer? { didSet { set(xyzHighBuffer, index: .Custom3) } }
    public var filesBuffer: MTLBuffer? { didSet { set(filesBuffer, index: .Custom4) } }
    public var pixelBuffer: MTLBuffer? { didSet { set(pixelBuffer, index: .Custom5) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y)))
        set("viewMatrix", viewMatrix)
        set("projectionMatrix", projectionMatrix)
        set("enableFrustumCulling", enableFrustumCulling ? 1 : 0)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: batchCount > 0 && screenSize.x > 0 && screenSize.y > 0 && batchesBuffer != nil && pixelBuffer != nil
        )
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreads(
            MTLSize(width: batchCount * computeRasteriserThreadsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: computeRasteriserThreadsPerGroup, height: 1, depth: 1)
        )
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: batchCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: computeRasteriserThreadsPerGroup, height: 1, depth: 1)
        )
    }
}

