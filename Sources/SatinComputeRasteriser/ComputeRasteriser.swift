import Metal
import Satin
import simd

public final class ComputeRasteriser: Object, @unchecked Sendable {
    public var configuration: ComputeRasteriserConfiguration = .init() {
        didSet { applyConfiguration() }
    }

    public nonisolated(unsafe) static var pipelinesURL: URL = {
        Bundle.module.resourceURL!.appendingPathComponent("Pipelines")
    }()

    public private(set) var outputTexture: MTLTexture?
    private var outputTextureB: MTLTexture?
    private var holeFillResultTexture: MTLTexture?

    private var pixelBuffer: MTLBuffer?
    private var nearestDepthBuffer: MTLBuffer?
    private var nearestIndexBuffer: MTLBuffer?
    private var viewport: SIMD4<Float> = .zero
    private var scaleFactor: Float = 1.0

    private lazy var clearProcessor = ClearProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var cullProcessor = CullProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var cullFinalizeProcessor = CullFinalizeProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var depthProcessor = DepthPassProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var colorProcessor = ColorPassProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var resolveProcessor = ResolveProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var clearWinnerProcessor = ClearWinnerProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestDepthProcessor = NearestDepthProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestIndexProcessor = NearestIndexProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var nearestResolveProcessor = NearestResolveProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)
    private lazy var holeFillProcessor = HoleFillProcessor(device: context.device, pipelinesURL: Self.pipelinesURL, live: true)

    private lazy var postMaterial: SourceMaterial = {
        let material = SourceMaterial(
            context: context,
            pipelineURL: Self.pipelinesURL.appendingPathComponent("Post.metal"),
            live: true
        )
        material.label = "ComputeRasteriserPost"
        material.lighting = false
        material.depthWriteEnabled = false
        material.depthCompareFunction = .always
        material.blending = .alpha
        return material
    }()

    private lazy var postProcessor = PostProcessor(
        label: "ComputeRasteriserPostProcessor",
        context: context,
        material: postMaterial
    )

    public init(context: Context, label: String = "ComputeRasteriser") {
        super.init(context: context, label: label)
    }

    public required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    @discardableResult
    public func addPointCloud(_ cloud: ComputeRasteriserPointCloud) -> ComputeRasteriserPointCloud {
        add(cloud)
        return cloud
    }

    public func removePointCloud(_ cloud: ComputeRasteriserPointCloud) {
        remove(cloud)
    }

    public var pointClouds: [ComputeRasteriserPointCloud] {
        children.compactMap { $0 as? ComputeRasteriserPointCloud }
    }

    public func resize(size: (width: Float, height: Float), scaleFactor: Float = 1.0) {
        let nextViewport = SIMD4<Float>(0, 0, size.width, size.height)
        guard nextViewport != viewport || scaleFactor != self.scaleFactor else { return }
        viewport = nextViewport
        self.scaleFactor = scaleFactor
        resizeResources()
    }

    public override func setup() {
        super.setup()
        applyConfiguration()
        if viewport.z > 0, viewport.w > 0 {
            resizeResources()
        }
    }

    public override func update(renderContext: Context, camera: Camera, viewport: simd_float4, index: Int) {
        guard visible else { return }
        if self.viewport.z != viewport.z || self.viewport.w != viewport.w {
            self.viewport = viewport
            resizeResources()
        }

        let viewProjection = camera.projectionMatrix * camera.viewMatrix
        let screenSize = SIMD2<UInt32>(UInt32(max(viewport.z, 0)), UInt32(max(viewport.w, 0)))

        cullProcessor.screenSize = screenSize
        cullProcessor.viewMatrix = camera.viewMatrix
        cullProcessor.projectionMatrix = camera.projectionMatrix
        cullProcessor.enableFrustumCulling = configuration.enableFrustumCulling
        cullProcessor.lodBias = configuration.lodBias

        for processor in [depthProcessor, nearestDepthProcessor, nearestIndexProcessor, colorProcessor] {
            processor.screenSize = screenSize
            processor.viewMatrix = camera.viewMatrix
            processor.projectionMatrix = camera.projectionMatrix
            processor.pointSizeMode = configuration.pointSizeMode
            processor.minimumPointSize = configuration.minimumPointSize
            processor.maximumPointSize = configuration.maximumPointSize
            processor.pointSizeScale = configuration.pointSizeScale
            processor.lodDither = configuration.enableLODDither
        }

        colorProcessor.depthTolerance = configuration.depthTolerance
        colorProcessor.colorizeChunks = configuration.colorizeChunks
        colorProcessor.colorizeOverdraw = configuration.colorizeOverdraw

        for cloud in pointClouds where cloud.visible {
            cloud.updateFiles(viewProjection: viewProjection, modelMatrix: cloud.worldMatrix)
        }
    }

    public override func encode(_ commandBuffer: MTLCommandBuffer) {
        guard visible,
              let pixelBuffer,
              outputTexture != nil,
              Int(viewport.z) > 0,
              Int(viewport.w) > 0
        else { return }

        switch configuration.mode {
        case .highQualityAverage:
            encodeHighQualityAverage(commandBuffer, pixelBuffer: pixelBuffer)
        case .nearestPoint:
            encodeNearestPoint(commandBuffer)
        }

        encodeHoleFill(commandBuffer)
    }

    private func encodeHoleFill(_ commandBuffer: MTLCommandBuffer) {
        let iterations = max(0, configuration.holeFillIterations)
        guard iterations > 0,
              let source = outputTexture,
              let target = outputTextureB
        else {
            holeFillResultTexture = outputTexture
            return
        }

        holeFillProcessor.width = Int(viewport.z)
        holeFillProcessor.height = Int(viewport.w)

        var src = source
        var dst = target
        for _ in 0 ..< iterations {
            holeFillProcessor.inputTexture = src
            holeFillProcessor.outputTexture = dst
            holeFillProcessor.update(commandBuffer)
            swap(&src, &dst)
        }
        holeFillResultTexture = src
    }

    private func encodeHighQualityAverage(_ commandBuffer: MTLCommandBuffer, pixelBuffer: MTLBuffer) {
        clearProcessor.pixelBuffer = pixelBuffer
        clearProcessor.pixelCount = Int(viewport.z) * Int(viewport.w)
        clearProcessor.update(commandBuffer)

        for cloud in pointClouds where cloud.visible && cloud.batchCount > 0 {
            guard runCullPass(commandBuffer, cloud: cloud) else { continue }

            bind(cloud, to: depthProcessor, pixelBuffer: pixelBuffer)
            depthProcessor.update(commandBuffer)

            bind(cloud, to: colorProcessor, pixelBuffer: pixelBuffer)
            colorProcessor.colorsBuffer = cloud.colorsBuffer
            colorProcessor.update(commandBuffer)
        }

        resolveProcessor.pixelBuffer = pixelBuffer
        resolveProcessor.outputTexture = outputTexture
        resolveProcessor.update(commandBuffer)
    }

    private func encodeNearestPoint(_ commandBuffer: MTLCommandBuffer) {
        guard let nearestDepthBuffer,
              let nearestIndexBuffer,
              let cloud = pointClouds.first(where: { $0.visible && $0.batchCount > 0 })
        else { return }

        clearWinnerProcessor.depthBuffer = nearestDepthBuffer
        clearWinnerProcessor.indexBuffer = nearestIndexBuffer
        clearWinnerProcessor.pixelCount = Int(viewport.z) * Int(viewport.w)
        clearWinnerProcessor.update(commandBuffer)

        guard runCullPass(commandBuffer, cloud: cloud) else { return }

        bind(cloud, to: nearestDepthProcessor, pixelBuffer: nearestDepthBuffer)
        nearestDepthProcessor.update(commandBuffer)

        bind(cloud, to: nearestIndexProcessor, pixelBuffer: nearestDepthBuffer)
        nearestIndexProcessor.indexBuffer = nearestIndexBuffer
        nearestIndexProcessor.update(commandBuffer)

        nearestResolveProcessor.depthBuffer = nearestDepthBuffer
        nearestResolveProcessor.indexBuffer = nearestIndexBuffer
        nearestResolveProcessor.colorsBuffer = cloud.colorsBuffer
        nearestResolveProcessor.outputTexture = outputTexture
        nearestResolveProcessor.update(commandBuffer)
    }

    private func runCullPass(_ commandBuffer: MTLCommandBuffer, cloud: ComputeRasteriserPointCloud) -> Bool {
        guard let visibleBuffer = cloud.visibleBatchesBuffer,
              let counterBuffer = cloud.cullCounterBuffer,
              let indirectArgsBuffer = cloud.cullIndirectArgsBuffer
        else { return false }

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "\(label).CullReset"
            blit.fill(buffer: counterBuffer, range: 0 ..< counterBuffer.length, value: 0)
            blit.fill(buffer: indirectArgsBuffer, range: 0 ..< indirectArgsBuffer.length, value: 0)
            blit.endEncoding()
        }

        cullProcessor.batchCount = cloud.batchCount
        cullProcessor.batchesBuffer = cloud.batchesBuffer
        cullProcessor.filesBuffer = cloud.filesBuffer
        cullProcessor.visibleBuffer = visibleBuffer
        cullProcessor.counterBuffer = counterBuffer
        cullProcessor.update(commandBuffer)

        cullFinalizeProcessor.counterBuffer = counterBuffer
        cullFinalizeProcessor.indirectArgsBuffer = indirectArgsBuffer
        cullFinalizeProcessor.update(commandBuffer)

        return true
    }

    public func draw(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        let finalTexture = holeFillResultTexture ?? outputTexture
        guard let finalTexture else { return }
        postMaterial.set(finalTexture, index: FragmentTextureIndex.Custom1)
        postProcessor.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
    }

    private func bind(_ cloud: ComputeRasteriserPointCloud, to processor: DepthPassProcessor, pixelBuffer: MTLBuffer) {
        processor.batchesBuffer = cloud.batchesBuffer
        processor.xyzLowBuffer = cloud.xyzLowBuffer
        processor.xyzMedBuffer = cloud.xyzMedBuffer
        processor.xyzHighBuffer = cloud.xyzHighBuffer
        processor.filesBuffer = cloud.filesBuffer
        processor.pixelBuffer = pixelBuffer
        processor.levelsBuffer = cloud.levelsBuffer
        processor.visibleBatchesBuffer = cloud.visibleBatchesBuffer
        processor.indirectArgsBuffer = cloud.cullIndirectArgsBuffer
    }

    private func resizeResources() {
        let width = Int(viewport.z)
        let height = Int(viewport.w)
        let pixelCount = width * height
        guard width > 0, height > 0, pixelCount > 0 else { return }

        pixelBuffer = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<RasterPixel>.stride,
            options: .storageModePrivate
        )
        pixelBuffer?.label = "\(label).Pixels"
        nearestDepthBuffer = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )
        nearestDepthBuffer?.label = "\(label).NearestDepths"
        nearestIndexBuffer = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )
        nearestIndexBuffer?.label = "\(label).NearestIndices"
        outputTexture = makeOutputTexture(width: width, height: height, label: "\(label).Output")
        outputTextureB = makeOutputTexture(width: width, height: height, label: "\(label).OutputB")
        holeFillResultTexture = outputTexture

        clearProcessor.pixelCount = pixelCount
        clearWinnerProcessor.pixelCount = pixelCount
        resolveProcessor.width = width
        resolveProcessor.height = height
        resolveProcessor.backgroundColor = configuration.backgroundColor
        nearestResolveProcessor.width = width
        nearestResolveProcessor.height = height
        nearestResolveProcessor.backgroundColor = configuration.backgroundColor
        postProcessor.resize(size: (Float(width), Float(height)), scaleFactor: scaleFactor)
    }

    private func applyConfiguration() {
        resolveProcessor.backgroundColor = configuration.backgroundColor
        nearestResolveProcessor.backgroundColor = configuration.backgroundColor
        cullProcessor.enableFrustumCulling = configuration.enableFrustumCulling
        cullProcessor.lodBias = configuration.lodBias
        for processor in [depthProcessor, nearestDepthProcessor, nearestIndexProcessor, colorProcessor] {
            processor.pointSizeMode = configuration.pointSizeMode
            processor.minimumPointSize = configuration.minimumPointSize
            processor.maximumPointSize = configuration.maximumPointSize
            processor.pointSizeScale = configuration.pointSizeScale
            processor.lodDither = configuration.enableLODDither
        }
        colorProcessor.depthTolerance = configuration.depthTolerance
        colorProcessor.colorizeChunks = configuration.colorizeChunks
        colorProcessor.colorizeOverdraw = configuration.colorizeOverdraw
    }

    private func makeOutputTexture(width: Int, height: Int, label: String) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        descriptor.storageMode = .private
        let texture = context.device.makeTexture(descriptor: descriptor)
        texture?.label = label
        return texture
    }
}
