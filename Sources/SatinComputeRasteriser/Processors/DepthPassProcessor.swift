import Metal
import Satin
import simd

open class DepthPassProcessor: BaseComputeRasteriserProcessor {
    public var screenSize: SIMD2<UInt32> = .zero {
        didSet { set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y))) }
    }

    public var viewMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("viewMatrix", viewMatrix) }
    }

    public var projectionMatrix: simd_float4x4 = matrix_identity_float4x4 {
        didSet { set("projectionMatrix", projectionMatrix) }
    }

    public var pointSizeMode: PointSizeMode = .screenSpace {
        didSet { set("pointSizeMode", Int(pointSizeMode.rawValue)) }
    }

    public var minimumPointSize: Float = 1.0 {
        didSet { set("minimumPointSize", minimumPointSize) }
    }

    public var maximumPointSize: Float = 1.0 {
        didSet { set("maximumPointSize", maximumPointSize) }
    }

    public var pointSizeScale: Float = 1.0 {
        didSet { set("pointSizeScale", pointSizeScale) }
    }

    public var batchesBuffer: MTLBuffer? { didSet { set(batchesBuffer, index: .Custom0) } }
    public var xyzLowBuffer: MTLBuffer? { didSet { set(xyzLowBuffer, index: .Custom1) } }
    public var xyzMedBuffer: MTLBuffer? { didSet { set(xyzMedBuffer, index: .Custom2) } }
    public var xyzHighBuffer: MTLBuffer? { didSet { set(xyzHighBuffer, index: .Custom3) } }
    public var filesBuffer: MTLBuffer? { didSet { set(filesBuffer, index: .Custom4) } }
    public var pixelBuffer: MTLBuffer? { didSet { set(pixelBuffer, index: .Custom5) } }
    public var visibleBatchesBuffer: MTLBuffer? { didSet { set(visibleBatchesBuffer, index: .Custom7) } }

    public var indirectArgsBuffer: MTLBuffer?
    public var indirectArgsBufferOffset: Int = 0

    open override func setup() {
        super.setup()
        set("screenSize", simd_int2(Int32(screenSize.x), Int32(screenSize.y)))
        set("viewMatrix", viewMatrix)
        set("projectionMatrix", projectionMatrix)
        set("pointSizeMode", Int(pointSizeMode.rawValue))
        set("minimumPointSize", minimumPointSize)
        set("maximumPointSize", maximumPointSize)
        set("pointSizeScale", pointSizeScale)
    }

    open override func update(_ commandBuffer: MTLCommandBuffer, iterations: Int = 1) {
        encodeIfReady(
            commandBuffer,
            isReady: screenSize.x > 0
                && screenSize.y > 0
                && batchesBuffer != nil
                && pixelBuffer != nil
                && visibleBatchesBuffer != nil
                && indirectArgsBuffer != nil
        )
    }

#if os(macOS) || os(iOS) || os(visionOS)
    open override func dispatchThreads(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        dispatchThreadgroups(computeEncoder: computeEncoder, pipeline: pipeline, iteration: iteration)
    }
#endif

    open override func dispatchThreadgroups(computeEncoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, iteration: Int) {
        guard let indirectArgsBuffer else { return }
        computeEncoder.dispatchThreadgroups(
            indirectBuffer: indirectArgsBuffer,
            indirectBufferOffset: indirectArgsBufferOffset,
            threadsPerThreadgroup: MTLSize(width: computeRasteriserThreadsPerGroup, height: 1, depth: 1)
        )
    }
}
