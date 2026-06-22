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
    /// Per-pixel reversed-Z NDC depth (R32Float; 0 = no cloud) written by the
    /// resolve pass. Sampled by the depth-aware composite so the cloud
    /// inter-occludes with regular Satin meshes. See ``ComputeRasteriserConfiguration/writesSceneDepth``.
    private var depthTexture: MTLTexture?

    private var pixelBuffer: MTLBuffer?
    private var nearestDepthBuffer: MTLBuffer?
    private var nearestIndexBuffer: MTLBuffer?

    // A single zeroed buffer bound as the displacement/tint stand-in for clouds
    // that have no real buffer, when `applyDisplacement`/`applyTint` is on.
    // Without it, those clouds would bind `xyzLowBuffer` (packed positions) and
    // read it as garbage offsets/colours — so a per-cloud pass on one cloud of a
    // multi-cloud mosaic corrupts the others. Sized to the largest cloud's point
    // capacity (reads for any cloud stay in-bounds); zeroed once on (re)alloc.
    private var zeroStandInBuffer: MTLBuffer?
    private var zeroStandInPoints: Int = 0
    private var currentStandIn: MTLBuffer?
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
        var depthTexture: MTLTexture
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

    // colorLoadAction defaults to `.clear` on PostProcessEncoder, which would
    // wipe the entire render-pass color attachment on every composite call.
    // That breaks stereo offline rendering (the right-eye composite clears the
    // left-eye composite) and means the rasteriser unconditionally replaces
    // any scene content below. `.load` makes it a true overlay: prior scene
    // pixels are preserved, and the postMaterial's `blending = .alpha` does
    // the actual composite.
    private lazy var postProcessor = PostProcessEncoder(
        label: "ComputeRasteriserPostProcessor",
        context: context,
        material: postMaterial,
        colorLoadAction: .load,
        depthLoadAction: .load
    )

    // Depth-aware variant of the composite. Label drives the shader function
    // names (`computeRasteriserPostDepthVertex/Fragment` in Post.metal). Tests
    // the cloud's per-pixel depth against the scene depth (`.greaterEqual`,
    // reversed-Z) so meshes inter-occlude with the cloud, AND writes that depth
    // back into the attachment so downstream depth consumers (a depth-of-field
    // post pass, etc.) see the cloud at its true distance — matching the
    // documented ``ComputeRasteriserConfiguration/writesSceneDepth`` behavior.
    // Used when `configuration.writesSceneDepth` is true.
    private lazy var postDepthMaterial: SourceMaterial = {
        let material = SourceMaterial(
            context: context,
            pipelineURL: Self.pipelinesURL.appendingPathComponent("Post.metal"),
            live: true
        )
        material.label = "ComputeRasteriserPostDepth"
        material.lighting = false
        material.depthWriteEnabled = true
        material.depthCompareFunction = .greaterEqual
        material.blending = .alpha
        return material
    }()

    private lazy var postDepthProcessor = PostProcessEncoder(
        label: "ComputeRasteriserPostDepthProcessor",
        context: context,
        material: postDepthMaterial,
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

    /// All point clouds in the rasteriser's subtree — not just direct children.
    /// This lets clouds be organised under intermediate group `Object`s (e.g. a
    /// per-capture / per-month parent node) while still being culled, packed,
    /// and drawn. Each cloud's `worldMatrix` (used at render time) already
    /// composes any parent-group transform, so nesting is positionally correct.
    public var pointClouds: [ComputeRasteriserPointCloud] {
        var out: [ComputeRasteriserPointCloud] = []
        func collect(_ object: Object) {
            for child in object.children {
                if let cloud = child as? ComputeRasteriserPointCloud { out.append(cloud) }
                collect(child)
            }
        }
        collect(self)
        return out
    }

    /// Point clouds in the subtree that are effectively visible — the cloud's
    /// own `visible` **and** every intermediate group node up to this rasteriser
    /// is `visible`. A hidden group node therefore hides all of its child
    /// clouds, matching Satin's tree-visibility semantics for meshes (the
    /// renderer skips an invisible object's whole subtree). Computed top-down so
    /// it needs no access to Satin's internal `parent` link.
    ///
    /// Drawing/culling is gated on this; residency (``pointClouds``) is not, so
    /// toggling a group's visibility is instant and never re-streams its clouds.
    public var visiblePointClouds: [ComputeRasteriserPointCloud] {
        var out: [ComputeRasteriserPointCloud] = []
        func collect(_ object: Object, ancestorsVisible: Bool) {
            for child in object.children {
                let chainVisible = ancestorsVisible && child.visible
                if let cloud = child as? ComputeRasteriserPointCloud, chainVisible {
                    out.append(cloud)
                }
                collect(child, ancestorsVisible: chainVisible)
            }
        }
        collect(self, ancestorsVisible: true)
        return out
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
        colorProcessor.applyTint = configuration.applyTint

        // Translucent defocus (weighted-blended OIT). All three passes must agree:
        // the depth pass stops defocused points claiming depth, the colour pass
        // weighted-blends every point with no depth test, and the resolve composites
        // Σ(c·α)/Σα with alpha 1−e^(−Σα). See ComputeRasteriserConfiguration.
        // tintAlphaIsCoverage for the mechanism + the overdraw-scaling perf cost.
        let coverage = configuration.applyTint && configuration.tintAlphaIsCoverage
        depthProcessor.applyTint = configuration.applyTint
        depthProcessor.tintAlphaIsCoverage = configuration.tintAlphaIsCoverage
        colorProcessor.tintAlphaIsCoverage = configuration.tintAlphaIsCoverage
        resolveProcessor.coverageEnabled = coverage

        colorProcessor.depthTolerance = configuration.depthTolerance
        colorProcessor.colorizeChunks = configuration.colorizeChunks
        colorProcessor.colorizeOverdraw = configuration.colorizeOverdraw

        for cloud in visiblePointClouds {
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

        // When displacement/tint is globally on, clouds without their own buffer
        // must read zeros (not xyzLow garbage). Prepare a shared zeroed stand-in
        // sized to the largest drawable cloud so every cloud's reads are in-bounds.
        currentStandIn = nil
        if configuration.applyDisplacement || configuration.applyTint {
            let maxPoints = visiblePointClouds
                .filter { $0.residentBatchCount > 0 }
                .map { $0.capacity.maxResidentPoints }
                .max() ?? 0
            if maxPoints > 0 {
                currentStandIn = zeroStandIn(forPoints: maxPoints, commandBuffer: commandBuffer)
            }
        }

        // Depth and color are split into two passes ACROSS all clouds, not
        // interleaved per cloud. The color pass only commits a fragment whose
        // depth is within tolerance of the pixel's winning depth, and those
        // colour adds are atomic and irreversible. If we ran depth+color per
        // cloud, an earlier-iterated (farther) cloud would commit its colour
        // before a later (nearer) cloud lowered the pixel depth — its colour
        // could not then be retracted, so occlusion would depend on load order.
        // Running every cloud's depth pass first settles the global nearest
        // depth per pixel; only then does any colour accumulate. Occlusion is
        // therefore a pure function of depth, independent of cloud order.
        //
        // The cull/visible/indirect-args buffers are per-cloud, so each cloud
        // keeps its phase-1 cull result for phase 2. Metal's automatic hazard
        // tracking on the shared (private) pixelBuffer serialises the two
        // phases — same guarantee the per-cloud depth→color ordering already
        // relied on. Pass counts are unchanged: cull+depth+color once each.
        var culledClouds: [ComputeRasteriserPointCloud] = []
        for cloud in visiblePointClouds where cloud.residentBatchCount > 0 {
            guard runCullPass(commandBuffer, cloud: cloud) else { continue }

            bind(cloud, to: depthProcessor, pixelBuffer: pixelBuffer)
            bindDisplacement(cloud, to: depthProcessor)
            bindTint(cloud, to: depthProcessor)
            depthProcessor.update(commandBuffer)

            culledClouds.append(cloud)
        }

        for cloud in culledClouds {
            bind(cloud, to: colorProcessor, pixelBuffer: pixelBuffer)
            bindDisplacement(cloud, to: colorProcessor)
            bindTint(cloud, to: colorProcessor)
            colorProcessor.colorsBuffer = cloud.colorsBuffer
            colorProcessor.update(commandBuffer)
        }

        resolveProcessor.pixelBuffer = pixelBuffer
        resolveProcessor.outputTexture = outputTexture
        resolveProcessor.depthTexture = depthTexture
        resolveProcessor.update(commandBuffer)
    }

    private func encodeNearestPoint(_ commandBuffer: MTLCommandBuffer) {
        guard let nearestDepthBuffer,
              let nearestIndexBuffer,
              let cloud = visiblePointClouds.first(where: { $0.residentBatchCount > 0 })
        else { return }

        guard encodeNearestIndexPass(commandBuffer, cloud: cloud) else { return }

        nearestResolveProcessor.depthBuffer = nearestDepthBuffer
        nearestResolveProcessor.indexBuffer = nearestIndexBuffer
        nearestResolveProcessor.colorsBuffer = cloud.colorsBuffer
        nearestResolveProcessor.outputTexture = outputTexture
        nearestResolveProcessor.depthTexture = depthTexture
        nearestResolveProcessor.update(commandBuffer)
    }

    /// Fill `nearestDepthBuffer` + `nearestIndexBuffer` for a single `cloud`:
    /// per pixel, the front-most resident point's depth and global packed
    /// `pointIndex` (`UInt32.max` where no point lands). Shared by the
    /// `.nearestPoint` render mode and ``pickPointIndex(atNDC:in:camera:)``.
    /// Writes only those two buffers — **no** resolve, so no output/depth
    /// texture is touched and it is safe to run as an off-screen pick.
    private func encodeNearestIndexPass(_ commandBuffer: MTLCommandBuffer, cloud: ComputeRasteriserPointCloud) -> Bool {
        guard let nearestDepthBuffer, let nearestIndexBuffer else { return false }

        clearWinnerProcessor.depthBuffer = nearestDepthBuffer
        clearWinnerProcessor.indexBuffer = nearestIndexBuffer
        clearWinnerProcessor.pixelCount = Int(viewport.z) * Int(viewport.w)
        clearWinnerProcessor.update(commandBuffer)

        guard runCullPass(commandBuffer, cloud: cloud) else { return false }

        bind(cloud, to: nearestDepthProcessor, pixelBuffer: nearestDepthBuffer)
        nearestDepthProcessor.update(commandBuffer)

        bind(cloud, to: nearestIndexProcessor, pixelBuffer: nearestDepthBuffer)
        nearestIndexProcessor.indexBuffer = nearestIndexBuffer
        nearestIndexProcessor.update(commandBuffer)
        return true
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

    // MARK: - Per-point picking

    /// Configure the cull + nearest-pass processors for `camera` at the current
    /// viewport, mirroring the subset of ``update(renderContext:camera:viewport:index:)``
    /// the nearest-index pass relies on. Lets ``pickPointIndex(atNDC:in:camera:)``
    /// run outside the live update/encode cycle.
    private func configureNearestProcessors(camera: Camera) {
        let screenSize = SIMD2<UInt32>(UInt32(max(viewport.z, 0)), UInt32(max(viewport.w, 0)))
        cullProcessor.screenSize = screenSize
        cullProcessor.viewMatrix = camera.viewMatrix
        cullProcessor.projectionMatrix = camera.projectionMatrix
        cullProcessor.enableFrustumCulling = configuration.enableFrustumCulling
        cullProcessor.lodBias = configuration.lodBias
        cullProcessor.enableCLOD = configuration.enableCLOD

        for processor in [nearestDepthProcessor, nearestIndexProcessor] {
            processor.screenSize = screenSize
            processor.viewMatrix = camera.viewMatrix
            processor.projectionMatrix = camera.projectionMatrix
            processor.pointSizeMode = configuration.pointSizeMode
            processor.minimumPointSize = configuration.minimumPointSize
            processor.maximumPointSize = configuration.maximumPointSize
            processor.pointSizeScale = configuration.pointSizeScale
            processor.lodDither = configuration.enableLODDither
        }
    }

    /// Pick the front-most resident point of `cloud` under a viewport location,
    /// for per-point selection.
    ///
    /// Runs the nearest-point pass (cull → nearest-depth → nearest-index) for
    /// just `cloud` on its own command buffer, then reads back the winning point
    /// for the pixel. The returned value is the **global packed point index** —
    /// the same `pointIndex` the rasteriser shaders, ``DisplacementPass`` and
    /// ``TintPass`` use, and the index into a ``PackedPointCloud``'s
    /// ``PackedPointCloud/sourceIndices`` (to recover the original source point).
    ///
    /// - Important: This encodes, commits, and **waits** on its own command
    ///   buffer, and shares the rasteriser's per-viewport nearest/cull buffers
    ///   with the live render. Call it between frames (e.g. from a click
    ///   handler), never from inside the render loop. It writes only the
    ///   nearest depth/index buffers — no output or depth texture is touched, so
    ///   the live frame's composited image is left intact.
    ///
    /// - Parameters:
    ///   - ndc: Normalised device coordinates of the pick location, x and y in
    ///     `[-1, 1]` with **y up** (`+1` = top of the viewport) — the same
    ///     convention as `Ray(camera:coordinate:)`.
    ///   - cloud: The point cloud to test (one of ``pointClouds``). Picking is
    ///     per-cloud so a mosaic resolves the right object.
    ///   - camera: The camera the cloud is currently rendered through.
    /// - Returns: The global packed point index under `ndc`, or `nil` if the
    ///   cloud has no resident points, the location is off-cloud, or resources
    ///   aren't allocated yet.
    public func pickPointIndex(
        atNDC ndc: SIMD2<Float>,
        in cloud: ComputeRasteriserPointCloud,
        camera: Camera,
        searchRadius: Int = 10
    ) -> UInt32? {
        let width = Int(viewport.z)
        let height = Int(viewport.w)
        guard width > 0, height > 0, cloud.residentBatchCount > 0, nearestIndexBuffer != nil else { return nil }

        // NDC (y-up) → buffer pixel. The nearest-index shader writes row 0 at the
        // top of the viewport (it flips Y), so flip here to match its addressing.
        let px = min(max(Int((ndc.x * 0.5 + 0.5) * Float(width)), 0), width - 1)
        let pyFromBottom = Int((ndc.y * 0.5 + 0.5) * Float(height))
        let py = min(max(height - 1 - pyFromBottom, 0), height - 1)

        configureNearestProcessors(camera: camera)
        cloud.updateFiles(viewProjection: camera.projectionMatrix * camera.viewMatrix, modelMatrix: cloud.worldMatrix)

        // Point clouds are sparse: the exact cursor pixel often has no rendered
        // point even when you click "on" the cloud. So read back a band of rows
        // around the cursor and return the nearest hit within `searchRadius` px.
        let stride = MemoryLayout<UInt32>.stride
        let r = max(0, searchRadius)
        let rowStart = max(0, py - r)
        let rowEnd = min(height - 1, py + r)
        let rowCount = rowEnd - rowStart + 1
        let bandLength = rowCount * width * stride

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return nil }
        commandBuffer.label = "\(label).Pick"
        guard encodeNearestIndexPass(commandBuffer, cloud: cloud),
              let nearestIndexBuffer,
              let staging = context.device.makeBuffer(length: bandLength, options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { return nil }

        blit.label = "\(label).PickReadback"
        blit.copy(from: nearestIndexBuffer, sourceOffset: rowStart * width * stride,
                  to: staging, destinationOffset: 0, size: bandLength)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Nearest non-sentinel pixel to the cursor within the circular window.
        let buf = staging.contents().bindMemory(to: UInt32.self, capacity: rowCount * width)
        var best: UInt32?
        var bestDist = Int.max
        let r2 = r * r
        for y in rowStart...rowEnd {
            let dy = y - py
            let rowBase = (y - rowStart) * width
            for x in max(0, px - r)...min(width - 1, px + r) {
                let dx = x - px
                let dist = dx * dx + dy * dy
                if dist > r2 || dist >= bestDist { continue }
                let v = buf[rowBase + x]
                if v != UInt32.max { bestDist = dist; best = v }
            }
        }
        return best
    }

    public func draw(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        let finalTexture = holeFillResultTexture ?? outputTexture
        guard let finalTexture else { return }
        if configuration.writesSceneDepth, let depthTexture {
            postDepthMaterial.set(finalTexture, index: FragmentTextureIndex.Custom1)
            postDepthMaterial.set(depthTexture, index: FragmentTextureIndex.Custom2)
            postDepthProcessor.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
        } else {
            postMaterial.set(finalTexture, index: FragmentTextureIndex.Custom1)
            postProcessor.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
        }
    }

    /// Variant that composites onto a specific sub-region of the render target,
    /// e.g. one eye's half viewport for stereo offline rendering. Forwards to
    /// Satin's `PostProcessEncoder.draw(viewports:)` which respects the supplied
    /// viewport instead of the post-processor's internal renderer viewport.
    public func draw(
        renderPassDescriptor: MTLRenderPassDescriptor,
        commandBuffer: MTLCommandBuffer,
        viewport: MTLViewport
    ) {
        let finalTexture = holeFillResultTexture ?? outputTexture
        guard let finalTexture else { return }
        if configuration.writesSceneDepth, let depthTexture {
            postDepthMaterial.set(finalTexture, index: FragmentTextureIndex.Custom1)
            postDepthMaterial.set(depthTexture, index: FragmentTextureIndex.Custom2)
            postDepthProcessor.draw(
                renderPassDescriptor: renderPassDescriptor,
                commandBuffer: commandBuffer,
                viewports: [viewport]
            )
        } else {
            postMaterial.set(finalTexture, index: FragmentTextureIndex.Custom1)
            postProcessor.draw(
                renderPassDescriptor: renderPassDescriptor,
                commandBuffer: commandBuffer,
                viewports: [viewport]
            )
        }
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
        processor.displacementBuffer = cloud.displacementBuffer ?? currentStandIn ?? cloud.xyzLowBuffer
    }

    /// Depth + Color passes both read tint at Custom10. Custom10 must always be
    /// bound for Metal validation — fall back to `xyzLowBuffer` when the user
    /// hasn't supplied a real tint buffer; the shaders gate reads on `applyTint`.
    /// Takes `DepthPassProcessor` so it serves both the depth and colour passes.
    private func bindTint(_ cloud: ComputeRasteriserPointCloud, to processor: DepthPassProcessor) {
        processor.tintBuffer = cloud.tintBuffer ?? currentStandIn ?? cloud.xyzLowBuffer
    }

    /// A shared zeroed buffer covering `points` (16 B/point — fits both float3
    /// displacement and float4 tint). Allocated + zeroed once, grown if a larger
    /// cloud appears. Lets a per-cloud Displacement/Tint pass affect only its own
    /// cloud; the rest read zeros instead of `xyzLow` garbage.
    private func zeroStandIn(forPoints points: Int, commandBuffer: MTLCommandBuffer) -> MTLBuffer? {
        if let buf = zeroStandInBuffer, zeroStandInPoints >= points { return buf }
        let stride = MemoryLayout<SIMD4<Float>>.stride
        let length = max(points, 1) * stride
        guard let buf = context.device.makeBuffer(length: length, options: .storageModePrivate) else {
            return zeroStandInBuffer
        }
        buf.label = "\(label).ZeroStandIn"
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "\(label).ZeroStandInFill"
            blit.fill(buffer: buf, range: 0 ..< length, value: 0)
            blit.endEncoding()
        }
        zeroStandInBuffer = buf
        zeroStandInPoints = max(points, 1)
        return buf
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
        depthTexture = resources.depthTexture
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
        postDepthProcessor.resize(size: (Float(width), Float(height)), scaleFactor: scaleFactor)
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
        let depth = makeDepthTexture(width: width, height: height, label: "\(label).Depth[\(width)x\(height)]")!

        let resources = CachedResources(
            pixelBuffer: pixel,
            nearestDepthBuffer: nearestDepth,
            nearestIndexBuffer: nearestIndex,
            outputTexture: outA,
            outputTextureB: outB,
            depthTexture: depth
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

    /// R32Float depth target: written by the resolve compute pass, sampled by
    /// the depth-aware composite. Not a render target (depth lives in the host
    /// render pass's own attachment), so usage is shader read/write only.
    private func makeDepthTexture(width: Int, height: Int, label: String) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        let texture = context.device.makeTexture(descriptor: descriptor)
        texture?.label = label
        return texture
    }
}
