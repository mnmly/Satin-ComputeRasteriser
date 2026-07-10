import Metal
import Satin
import simd

open class CullProcessor: BaseComputeRasteriserProcessor {
    /// Number of batch slots the cull kernel scans, and the dispatch width.
    /// Per-cloud, so it is bound per dispatch via `setBytes` (Custom4) in
    /// ``applyAdditionalBindings(_:)`` — never through the uniform struct,
    /// whose single per-frame slot is shared by every cloud's dispatch.
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

    public var lodBias: Int = 0 {
        didSet { set("lodBias", lodBias) }
    }

    /// CLOD master switch. When `false`, the shader writes a sentinel
    /// `lodThreshold` so downstream passes don't cull any points by level.
    public var enableCLOD: Bool = true {
        didSet { set("enableCLOD", enableCLOD ? 1 : 0) }
    }

    public var batchesBuffer: MTLBuffer? { didSet { set(batchesBuffer, index: .Custom0) } }
    public var filesBuffer: MTLBuffer? { didSet { set(filesBuffer, index: .Custom1) } }
    /// Byte offset into ``filesBuffer`` for the current ring slot. Bound on
    /// top of the standard `MTLBuffer` binding (which uses offset 0) via
    /// ``applyAdditionalBindings(_:)``.
    public var filesBufferOffset: Int = 0

    open override func applyAdditionalBindings(_ computeEncoder: MTLComputeCommandEncoder) {
        if let filesBuffer {
            computeEncoder.setBuffer(filesBuffer, offset: filesBufferOffset, index: ComputeBufferIndex.Custom1.rawValue)
        }
        var count = Int32(batchCount)
        computeEncoder.setBytes(&count, length: MemoryLayout<Int32>.stride, index: ComputeBufferIndex.Custom4.rawValue)
    }

    public var visibleBuffer: MTLBuffer? { didSet { set(visibleBuffer, index: .Custom2) } }
    public var counterBuffer: MTLBuffer? { didSet { set(counterBuffer, index: .Custom3) } }

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y)))
        set("viewMatrix", viewMatrix)
        set("projectionMatrix", projectionMatrix)
        set("enableFrustumCulling", enableFrustumCulling ? 1 : 0)
        set("lodBias", lodBias)
        set("enableCLOD", enableCLOD ? 1 : 0)
    }

    var isEncodeReady: Bool {
        batchCount > 0
            && batchesBuffer != nil
            && filesBuffer != nil
            && visibleBuffer != nil
            && counterBuffer != nil
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(commandBuffer, isReady: isEncodeReady)
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
