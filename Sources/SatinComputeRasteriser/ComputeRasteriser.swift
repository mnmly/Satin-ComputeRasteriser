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

    /// Cached per-viewport GPU resources. Offline recording alternates between
    /// the live drawable and the offline target sizes every frame; without a
    /// cache, each switch reallocates ~`width*height*48 B` for pixelBuffer +
    /// two RGBA8 output textures (≈200 MB for a 4K target). The cache is
    /// keyed by integer pixel size and capped — `resizeKeys` tracks LRU order.
    private struct CachedResources {
        var pixelBuffer: MTLBuffer
        var nearestDepthBuffer: MTLBuffer
        var nearestIndexBuffer: MTLBuffer
        var outputTexture: MTLTexture
        var outputTextureB: MTLTexture
    }
    private var resourceCache: [SIMD2<Int32>: CachedResources] = [:]
    private var resourceCacheLRU: [SIMD2<Int32>] = []
    private static let resourceCacheCap = 4

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

    // colorLoadAction defaults to `.clear` on PostProcessor, which would wipe
    // the entire render-pass color attachment on every composite call. That
    // breaks stereo offline rendering (the right-eye composite clears the
    // left-eye composite) and means the rasteriser unconditionally replaces
    // any scene content below. `.load` makes it a true overlay: prior scene
    // pixels are preserved, and the postMaterial's `blending = .alpha` does
    // the actual composite.
    private lazy var postProcessor = PostProcessor(
        label: "ComputeRasteriserPostProcessor",
        context: context,
        material: postMaterial,
        colorLoadAction: .load,
        depthLoadAction: .load
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
        cullProcessor.enableCLOD = configuration.enableCLOD

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

        depthProcessor.applyDisplacement = configuration.applyDisplacement
        colorProcessor.applyDisplacement = configuration.applyDisplacement

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

        for cloud in pointClouds where cloud.visible && cloud.residentBatchCount > 0 {
            guard runCullPass(commandBuffer, cloud: cloud) else { continue }

            bind(cloud, to: depthProcessor, pixelBuffer: pixelBuffer)
            bindDisplacement(cloud, to: depthProcessor)
            depthProcessor.update(commandBuffer)

            bind(cloud, to: colorProcessor, pixelBuffer: pixelBuffer)
            bindDisplacement(cloud, to: colorProcessor)
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
              let cloud = pointClouds.first(where: { $0.visible && $0.residentBatchCount > 0 })
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
        cullProcessor.filesBufferOffset = cloud.filesBufferOffset
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

    /// Variant that composites onto a specific sub-region of the render target,
    /// e.g. one eye's half viewport for stereo offline rendering. Forwards to
    /// Satin's `PostProcessor.draw(viewports:)` which respects the supplied
    /// viewport instead of the post-processor's internal renderer viewport.
    public func draw(
        renderPassDescriptor: MTLRenderPassDescriptor,
        commandBuffer: MTLCommandBuffer,
        viewport: MTLViewport
    ) {
        let finalTexture = holeFillResultTexture ?? outputTexture
        guard let finalTexture else { return }
        postMaterial.set(finalTexture, index: FragmentTextureIndex.Custom1)
        postProcessor.draw(
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            viewports: [viewport]
        )
    }

    private func bind(_ cloud: ComputeRasteriserPointCloud, to processor: DepthPassProcessor, pixelBuffer: MTLBuffer) {
        processor.batchesBuffer = cloud.batchesBuffer
        processor.xyzLowBuffer = cloud.xyzLowBuffer
        processor.xyzMedBuffer = cloud.xyzMedBuffer
        processor.xyzHighBuffer = cloud.xyzHighBuffer
        processor.filesBuffer = cloud.filesBuffer
        processor.filesBufferOffset = cloud.filesBufferOffset
        processor.pixelBuffer = pixelBuffer
        processor.levelsBuffer = cloud.levelsBuffer
        processor.visibleBatchesBuffer = cloud.visibleBatchesBuffer
        processor.indirectArgsBuffer = cloud.cullIndirectArgsBuffer
    }

    /// DepthPass + ColorPass both read displacement at Custom8. NearestDepth /
    /// NearestIndex use Custom8 for other data, so we set displacement only on
    /// the two HQ-average processors. Custom8 must always be bound for Metal
    /// validation — fall back to `xyzLowBuffer` when the user hasn't supplied a
    /// real displacement buffer; the shader gates reads on `applyDisplacement`.
    private func bindDisplacement(_ cloud: ComputeRasteriserPointCloud, to processor: DepthPassProcessor) {
        processor.displacementBuffer = cloud.displacementBuffer ?? cloud.xyzLowBuffer
    }

    private func resizeResources() {
        let width = Int(viewport.z)
        let height = Int(viewport.w)
        let pixelCount = width * height
        guard width > 0, height > 0, pixelCount > 0 else { return }

        let key = SIMD2<Int32>(Int32(width), Int32(height))
        let resources = resourceCache[key] ?? allocateAndCache(width: width, height: height, key: key)

        // Promote to most-recently-used.
        if let idx = resourceCacheLRU.firstIndex(of: key) {
            resourceCacheLRU.remove(at: idx)
        }
        resourceCacheLRU.append(key)

        pixelBuffer = resources.pixelBuffer
        nearestDepthBuffer = resources.nearestDepthBuffer
        nearestIndexBuffer = resources.nearestIndexBuffer
        outputTexture = resources.outputTexture
        outputTextureB = resources.outputTextureB
        holeFillResultTexture = resources.outputTexture

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

    private func allocateAndCache(width: Int, height: Int, key: SIMD2<Int32>) -> CachedResources {
        let pixelCount = width * height
        let pixel = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<RasterPixel>.stride,
            options: .storageModePrivate
        )!
        pixel.label = "\(label).Pixels[\(width)x\(height)]"
        let nearestDepth = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        nearestDepth.label = "\(label).NearestDepths[\(width)x\(height)]"
        let nearestIndex = context.device.makeBuffer(
            length: pixelCount * MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )!
        nearestIndex.label = "\(label).NearestIndices[\(width)x\(height)]"
        let outA = makeOutputTexture(width: width, height: height, label: "\(label).Output[\(width)x\(height)]")!
        let outB = makeOutputTexture(width: width, height: height, label: "\(label).OutputB[\(width)x\(height)]")!

        let resources = CachedResources(
            pixelBuffer: pixel,
            nearestDepthBuffer: nearestDepth,
            nearestIndexBuffer: nearestIndex,
            outputTexture: outA,
            outputTextureB: outB
        )
        resourceCache[key] = resources

        // Evict oldest if over cap.
        while resourceCacheLRU.count >= Self.resourceCacheCap {
            let evict = resourceCacheLRU.removeFirst()
            resourceCache.removeValue(forKey: evict)
        }
        return resources
    }

    private func applyConfiguration() {
        resolveProcessor.backgroundColor = configuration.backgroundColor
        nearestResolveProcessor.backgroundColor = configuration.backgroundColor
        cullProcessor.enableFrustumCulling = configuration.enableFrustumCulling
        cullProcessor.lodBias = configuration.lodBias
        cullProcessor.enableCLOD = configuration.enableCLOD
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
