import Combine
import Metal
import Satin
import SatinComputeRasteriser
import simd
#if canImport(SwiftPDAL)
import SwiftPDAL
import SatinComputeRasteriserStreaming
#endif

/// Mirror of SwiftPDAL.ResidencyPolicy that doesn't require importing
/// SwiftPDAL into the AppState (which must compile without the dep).
public enum StreamingResidencyChoice: String, CaseIterable, Hashable {
    /// SwiftPDAL's `frustumFirstThenHalo` — frustum-visible chunks first,
    /// then surround the camera with leftover budget.
    case halo
    /// SwiftPDAL's `distanceOnly` — no frustum gate, fill nearest-first.
    /// Matches the "load it all" feel of non-streaming renderers.
    case distance
}

public final class ComputeRasteriserAppState: ObservableObject {
    @Published public var status: String = "Fixture"
    @Published public var errorMessage: String?
    @Published public var mode: ComputeRasteriserMode = .highQualityAverage
    @Published public var pointSizeMode: PointSizeMode = .screenSpace
    @Published public var minimumPointSize: Float = 1.0
    @Published public var maximumPointSize: Float = 5.0
    @Published public var pointSizeScale: Float = 5.0
    @Published public var lodBias: Int = 0
    @Published public var enableFrustumCulling: Bool = true
    @Published public var colorizeChunks: Bool = false
    @Published public var holeFillIterations: Int = 0
    @Published public var enableCLOD: Bool = true
    @Published public var enableLODDither: Bool = true
    /// Streaming-mode telemetry. Updated by the StreamingAdapter each frame.
    /// Zero in non-streaming (PLY/fixture) mode.
    @Published public var streamingChunks: Int = 0
    @Published public var streamingPoints: Int = 0
    /// Streaming budget in MB; the adapter passes this to the source's
    /// residency decider to cap working-set bytes.
    @Published public var streamingBudgetMB: Int = 1024
    @Published public var isStreaming: Bool = false
    @Published public var streamingResidency: StreamingResidencyChoice = .halo

    // Depth of field (weighted-blended-OIT translucent defocus + jitter spread).
    // The focus band is a FRACTION of the focal distance, so it auto-scales to
    // whatever cloud is loaded. See ComputeRasteriserConfiguration.tintAlphaIsCoverage.
    @Published public var dofEnabled: Bool = false
    @Published public var dofTranslucent: Bool = true   // see-through defocus (OIT)
    @Published public var dofJitter: Bool = true         // scatter spread (displacement)
    @Published public var dofAutoFocus: Bool = true      // focus on the cloud centre
    @Published public var dofFocus: Float = 1.0          // manual focal distance (world units)
    @Published public var dofBand: Float = 0.04          // sharp half-band (fraction of focal dist)
    @Published public var dofFalloff: Float = 0.25       // ramp to full effect (fraction)
    @Published public var dofScatter: Float = 0.05       // jitter spread (fraction)
    @Published public var dofMaxDefocus: Float = 0.85    // transparency cap (1 = can vanish)
    @Published public var dofFocusMax: Float = 10        // manual-focus slider upper bound

    public init() {}
}

open class ComputeRasteriserAppRenderer: ViewRenderer, @unchecked Sendable {
    public lazy var renderer = RenderEncoder(context: defaultContext)
    public lazy var rasteriser = ComputeRasteriser(context: defaultContext)
    public private(set) lazy var pointCloud = ComputeRasteriserPointCloud(
        context: defaultContext,
        packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 28)
    )
    public lazy var scene = Object(context: defaultContext, label: "Compute Rasteriser App", [rasteriser])

    private var currentViewport: SIMD2<Float> = SIMD2(800, 600)

    #if canImport(SwiftPDAL)
    private var streamingAdapter: StreamingAdapter?
    private var lastCOPCURL: URL?
    #endif
    public lazy var camera = PerspectiveCamera(
        context: defaultContext,
        position: [0.0, 0.0, 2.4],
        near: 0.01,
        far: 100.0,
        fov: 45.0
    )
    public private(set) var cameraController: PerspectiveCameraController?
    public let appState: ComputeRasteriserAppState
    private let initialPLYURL: URL?

    // Depth-of-field passes (created in setup from embedded kernels).
    private var displacementPass: DisplacementPass?
    private var tintPass: TintPass?
    private var cloudCenter: SIMD3<Float> = .zero

    // Byte-compatible with the embedded .metal `CameraUniforms` / `DofParams`.
    private struct DofCameraUniforms {
        var modelView: simd_float4x4 = matrix_identity_float4x4
        var near: Float = 0.01
        var far: Float = 100
        var focalDistance: Float = 1
    }
    private struct DofParams {
        var band: Float = 0.04
        var falloff: Float = 0.25
        var scatter: Float = 0.05
        var maxDefocus: Float = 0.85
    }

    public init(
        initialPLYURL: URL? = nil,
        initialMode: ComputeRasteriserMode = .highQualityAverage,
        appState: ComputeRasteriserAppState = ComputeRasteriserAppState()
    ) {
        self.initialPLYURL = initialPLYURL
        self.appState = appState
        self.appState.mode = initialMode
        let device = MTLCreateSystemDefaultDevice()!
        let context = Context(
            device: device,
            sampleCount: 1,
            colorPixelFormat: .bgra8Unorm,
            depthPixelFormat: .depth32Float
        )
        super.init(context: context)
    }

    open override func setup() {
        renderer.setClearColor([0.025, 0.028, 0.034, 1.0])
        rasteriser.addPointCloud(pointCloud)
        rasteriser.configuration.backgroundColor = [0, 0, 0, 0]
        rasteriser.configuration.mode = appState.mode
        rasteriser.configuration.pointSizeMode = appState.pointSizeMode
        rasteriser.configuration.minimumPointSize = appState.minimumPointSize
        rasteriser.configuration.maximumPointSize = appState.maximumPointSize
        rasteriser.configuration.pointSizeScale = appState.pointSizeScale
        rasteriser.configuration.lodBias = appState.lodBias
        rasteriser.configuration.enableFrustumCulling = appState.enableFrustumCulling
        rasteriser.configuration.colorizeChunks = appState.colorizeChunks
        rasteriser.configuration.holeFillIterations = appState.holeFillIterations
        rasteriser.configuration.enableCLOD = appState.enableCLOD
        rasteriser.configuration.enableLODDither = appState.enableLODDither
        camera.lookAt(target: .zero)
        cameraController = PerspectiveCameraController(camera: camera, view: metalView)
        cameraController?.defaultDistance = 2.4
        cameraController?.enable()

        makeDofPasses()

        if let initialPLYURL {
            loadPLY(url: initialPLYURL)
        }
    }

    open override func update() {
        cameraController?.update()
        #if canImport(SwiftPDAL)
        if let adapter = streamingAdapter {
            adapter.update(camera: camera, viewport: currentViewport)
            DispatchQueue.main.async { [appState, chunks = adapter.residentChunks, points = adapter.residentPoints] in
                appState.streamingChunks = chunks
                appState.streamingPoints = points
            }
        }
        #endif
    }

    open override func cleanup() {
        cameraController?.disable()
        cameraController = nil
        super.cleanup()
    }

    open override func draw(renderPassDescriptor: MTLRenderPassDescriptor, commandBuffer: MTLCommandBuffer) {
        encodeDof(commandBuffer: commandBuffer)
        renderer.draw(
            renderPassDescriptor: renderPassDescriptor,
            commandBuffer: commandBuffer,
            scene: scene,
            camera: camera
        )
        rasteriser.draw(renderPassDescriptor: renderPassDescriptor, commandBuffer: commandBuffer)
    }

    open override func resize(size: (width: Float, height: Float), scaleFactor: Float) {
        camera.aspect = size.width / size.height
        cameraController?.resize(size)
        renderer.resize(size)
        rasteriser.resize(size: size, scaleFactor: scaleFactor)
        currentViewport = SIMD2(size.width, size.height)
    }

    public func loadPLY(url: URL) {
        do {
            let shouldStop = url.startAccessingSecurityScopedResource()
            defer {
                if shouldStop {
                    url.stopAccessingSecurityScopedResource()
                }
            }

            let packed = try PLYPointCloudLoader.load(url: url)
            pointCloud.replacePackedPointCloud(packed)
            frameCamera(to: packed)
            DispatchQueue.main.async { [appState] in
                appState.status = url.lastPathComponent
                appState.errorMessage = nil
            }
        } catch {
            DispatchQueue.main.async { [appState] in
                appState.errorMessage = error.localizedDescription
            }
        }
    }

    #if canImport(SwiftPDAL)
    /// Open a COPC LAZ file as a streaming source, swap the rendered cloud,
    /// and frame the camera to its bounds. The previous (PLY/fixture) cloud
    /// is removed from the rasteriser.
    public func loadCOPC(url: URL) {
        let budgetBytes = max(64, appState.streamingBudgetMB) * 1024 * 1024
        Task.detached { [weak self] in
            guard let self else { return }
            let shouldStop = url.startAccessingSecurityScopedResource()
            defer {
                if shouldStop { url.stopAccessingSecurityScopedResource() }
            }
            do {
                // 1.2.0: parallel decode via reader pool + faster ticks.
                // decodeConcurrency = active core count; LAZ decompress is
                // single-threaded per chunk so going past cores doesn't help.
                let cores = max(2, ProcessInfo.processInfo.activeProcessorCount)
                let policy: SwiftPDAL.ResidencyPolicy =
                    (await MainActor.run { self.appState.streamingResidency }) == .distance
                    ? .distanceOnly : .frustumFirstThenHalo
                let opts = StreamingOptions(
                    maxInFlightLoads: cores * 2,
                    decodeConcurrency: cores,
                    driverTickInterval: .milliseconds(16),
                    residencyPolicy: policy
                )
                let source = try await SwiftPDAL.CopcStreamingPointCloudSource.open(url, options: opts)
                source.setBudget(budgetBytes)
                await MainActor.run {
                    self.lastCOPCURL = url
                    self.installStreamingSource(source, url: url, budgetBytes: budgetBytes)
                }
            } catch {
                await MainActor.run {
                    self.appState.errorMessage = "COPC open failed: \(error.localizedDescription)"
                }
            }
        }
    }

    public func setStreamingBudget(MB: Int) {
        let bytes = max(64, MB) * 1024 * 1024
        DispatchQueue.main.async { [appState] in appState.streamingBudgetMB = MB }
        streamingAdapter?.setBudget(bytes: bytes)
    }

    /// Switching residency policy requires re-opening the source — the
    /// driver reads the policy at construction. Re-opens the same URL with
    /// the new choice; ~100 ms hiccup while the hierarchy is rescanned.
    public func setResidency(_ choice: StreamingResidencyChoice) {
        DispatchQueue.main.async { [appState] in appState.streamingResidency = choice }
        if let url = lastCOPCURL {
            loadCOPC(url: url)
        }
    }

    @MainActor
    private func installStreamingSource(_ source: SwiftPDAL.CopcStreamingPointCloudSource, url: URL, budgetBytes: Int) {
        // Tear down the existing cloud + adapter.
        streamingAdapter?.close()
        streamingAdapter = nil
        rasteriser.removePointCloud(pointCloud)

        // Pool capacity: cap by budget so we don't oversubscribe VRAM. 17 B/point
        // × pointsPerBatch is the per-slot byte cost. Hard cap at 65536 slots
        // (~11 GB at the SwiftPDAL default 10240 pts/batch) to keep the cull
        // dispatch's per-frame threadgroup count bounded — every slot gets a
        // threadgroup whether resident or not.
        let pointsPerBatch = source.info.pointsPerBatch
        let bytesPerSlot = pointsPerBatch * 17
        let slotsByBudget = max(1, budgetBytes / max(1, bytesPerSlot))
        let cap = ComputeRasteriserCapacity(
            maxResidentBatches: min(slotsByBudget, 65536),
            pointsPerBatch: pointsPerBatch
        )

        // The source pre-shifts every chunk's positions by `info.originShift`
        // so they're small FP32-safe values centered near origin. We deliberately
        // do NOT bake originShift back into RasterFile.world — that would
        // re-translate to absolute world coords (e.g. ~640K for Autzen's Oregon
        // State Plane), and the subsequent `viewMatrix * world` would combine
        // two huge translations and lose ~6 decimal digits of precision per
        // axis. Render in shifted space; frame the camera there too.
        let originShiftF = SIMD3<Float>(
            Float(source.info.originShift.x),
            Float(source.info.originShift.y),
            Float(source.info.originShift.z)
        )
        let cloud = ComputeRasteriserPointCloud(
            context: defaultContext,
            capacity: cap,
            label: "ComputeRasteriserPointCloud.Streaming"
        )
        pointCloud = cloud
        rasteriser.addPointCloud(cloud)

        let adapter = StreamingAdapter(source: source, cloud: cloud)
        streamingAdapter = adapter

        let shiftedMin = source.info.bounds.min - originShiftF
        let shiftedMax = source.info.bounds.max - originShiftF
        frameCamera(toBoundsMin: shiftedMin, boundsMax: shiftedMax)
        appState.status = url.lastPathComponent
        appState.errorMessage = nil
        appState.isStreaming = true
        appState.streamingChunks = 0
        appState.streamingPoints = 0
    }
    #endif

    public func setLODBias(_ bias: Int) {
        rasteriser.configuration.lodBias = bias
        DispatchQueue.main.async { [appState] in
            appState.lodBias = bias
        }
    }

    public func setFrustumCulling(_ enabled: Bool) {
        rasteriser.configuration.enableFrustumCulling = enabled
        DispatchQueue.main.async { [appState] in
            appState.enableFrustumCulling = enabled
        }
    }

    public func setColorizeChunks(_ enabled: Bool) {
        rasteriser.configuration.colorizeChunks = enabled
        DispatchQueue.main.async { [appState] in
            appState.colorizeChunks = enabled
        }
    }

    public func setHoleFillIterations(_ iterations: Int) {
        rasteriser.configuration.holeFillIterations = iterations
        DispatchQueue.main.async { [appState] in
            appState.holeFillIterations = iterations
        }
    }

    public func setLODDither(_ enabled: Bool) {
        rasteriser.configuration.enableLODDither = enabled
        DispatchQueue.main.async { [appState] in
            appState.enableLODDither = enabled
        }
    }

    public func setCLOD(_ enabled: Bool) {
        rasteriser.configuration.enableCLOD = enabled
        DispatchQueue.main.async { [appState] in
            appState.enableCLOD = enabled
        }
    }

    public func setMode(_ mode: ComputeRasteriserMode) {
        rasteriser.configuration.mode = mode
        DispatchQueue.main.async { [appState] in
            appState.mode = mode
        }
    }

    public func setPointSizing(mode: PointSizeMode? = nil, minimum: Float? = nil, maximum: Float? = nil, scale: Float? = nil) {
        if let mode {
            if mode != rasteriser.configuration.pointSizeMode {
                rasteriser.configuration.pointSizeScale = mode == .worldSpace ? 0.01 : 5.0
                rasteriser.configuration.maximumPointSize = mode == .worldSpace ? 64.0 : 5.0
            }
            rasteriser.configuration.pointSizeMode = mode
        }
        if let minimum {
            rasteriser.configuration.minimumPointSize = minimum
        }
        if let maximum {
            rasteriser.configuration.maximumPointSize = maximum
        }
        if let scale {
            rasteriser.configuration.pointSizeScale = scale
        }

        DispatchQueue.main.async { [appState, configuration = rasteriser.configuration] in
            appState.pointSizeMode = configuration.pointSizeMode
            appState.minimumPointSize = configuration.minimumPointSize
            appState.maximumPointSize = configuration.maximumPointSize
            appState.pointSizeScale = configuration.pointSizeScale
        }
    }

    private func frameCamera(to packed: PackedPointCloud) {
        frameCamera(toBoundsMin: packed.boundsMin, boundsMax: packed.boundsMax)
    }

    private func frameCamera(toBoundsMin boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>) {
        let center = (boundsMin + boundsMax) * 0.5
        let extent = boundsMax - boundsMin
        let radius = max(simd_length(extent) * 0.5, 0.01)
        let distance = radius / max(tan(camera.fov * 0.5 * .pi / 180.0), 0.001)
        let position = center + SIMD3<Float>(0.0, 0.0, distance * 1.35)

        // DoF: focus on the cloud centre by default, size the manual slider to fit.
        cloudCenter = center
        let camDist = simd_length(position - center)
        DispatchQueue.main.async { [appState] in
            appState.dofFocus = camDist
            appState.dofFocusMax = max(camDist * 3, radius * 6)
        }

        cameraController?.disable()
        camera.position = position
        camera.near = max(distance - radius * 3.0, 0.001)
        camera.far = distance + radius * 4.0
        camera.lookAt(target: center)
        cameraController?.defaultDistance = simd_length(position - center)
        cameraController?.defaultPosition = camera.position
        cameraController?.defaultOrientation = camera.orientation
        cameraController?.enable()
    }
}

// MARK: - Depth of field

extension ComputeRasteriserAppRenderer {
    /// Build the DoF passes from the embedded kernels. The displacement pass
    /// scatters out-of-focus points; the tint pass (in coverage / weighted-blended
    /// OIT mode) makes them see-through so they blend instead of hard-occluding.
    func makeDofPasses() {
        guard let jURL = writeTempKernel(Self.dofJitterKernel, name: "DofJitter"),
              let tURL = writeTempKernel(Self.dofTranslucencyKernel, name: "DofTranslucency") else { return }
        let dp = DisplacementPass(rasteriser: rasteriser, kernelURL: jURL, live: false)
        dp.bindUserBuffers = { [weak self] enc in self?.bindDof(enc, user0: DisplacementPass.bufferUser0) }
        displacementPass = dp
        let tp = TintPass(rasteriser: rasteriser, kernelURL: tURL, live: false)
        tp.alphaIsCoverage = true   // translucent defocus (OIT), not a colour mix
        tp.bindUserBuffers = { [weak self] enc in self?.bindDof(enc, user0: TintPass.bufferUser0) }
        tintPass = tp
    }

    /// Encode the DoF passes before the rasteriser draws (so the colour pass sees
    /// this frame's displacement + tint). Disables a pass when its toggle is off.
    func encodeDof(commandBuffer: MTLCommandBuffer) {
        guard appState.dofEnabled else {
            displacementPass?.disable(); tintPass?.disable(); return
        }
        if appState.dofJitter { displacementPass?.encode(commandBuffer: commandBuffer, cloud: pointCloud) }
        else { displacementPass?.disable() }
        if appState.dofTranslucent { tintPass?.encode(commandBuffer: commandBuffer, cloud: pointCloud) }
        else { tintPass?.disable() }
    }

    /// Bind USER0 = per-cloud camera (modelView + focal distance), USER1 = the DoF
    /// params, for whichever pass is encoding. The focus band is a fraction of the
    /// focal distance, so it auto-scales to the loaded cloud.
    private func bindDof(_ enc: MTLComputeCommandEncoder, user0: Int) {
        let fileWorld = pointCloud.files.first?.world ?? matrix_identity_float4x4
        let world = pointCloud.worldMatrix * fileWorld
        let focus = appState.dofAutoFocus ? simd_length(camera.worldPosition - cloudCenter) : appState.dofFocus
        var cam = DofCameraUniforms(
            modelView: camera.viewMatrix * world,
            near: camera.near, far: camera.far, focalDistance: max(focus, 1e-3)
        )
        withUnsafeBytes(of: &cam) { enc.setBytes($0.baseAddress!, length: $0.count, index: user0) }
        var dof = DofParams(
            band: appState.dofBand, falloff: appState.dofFalloff,
            scatter: appState.dofJitter ? appState.dofScatter : 0,
            maxDefocus: appState.dofTranslucent ? appState.dofMaxDefocus : 0
        )
        withUnsafeBytes(of: &dof) { enc.setBytes($0.baseAddress!, length: $0.count, index: user0 + 1) }
    }

    private func writeTempKernel(_ source: String, name: String) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name).metal")
        do {
            try source.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            DispatchQueue.main.async { [appState] in
                appState.errorMessage = "DoF kernel write failed: \(error.localizedDescription)"
            }
            return nil
        }
    }

    // USER0 = camera, USER1 = DoF params. SCR prepends its preamble before compile.
    private static let dofStructs = """
    typedef struct {
        float4x4 modelView;     // camera.view · cloud.world  (decoded-local → view)
        float    near;
        float    far;
        float    focalDistance; // sharp distance from the camera (view-space units)
    } CameraUniforms;
    typedef struct {
        float band;       // sharp half-band, fraction of focal distance
        float falloff;    // ramp to full effect, fraction of focal distance
        float scatter;    // jitter spread, fraction of focal distance
        float maxDefocus; // transparency cap (1 = can fully vanish)
    } DofParams;
    inline float dofHash1(uint x) {
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return float(x) * (1.0 / 4294967296.0);
    }
    inline float3 dofHash3(uint i, uint salt) {
        return float3(dofHash1(i * 747796405u  + salt),
                      dofHash1(i * 2891336453u + salt + 17u),
                      dofHash1(i * 3266489917u + salt + 101u)) * 2.0 - 1.0;
    }
    """

    static let dofJitterKernel = dofStructs + """

    kernel void computeDisplacement(
        SCR_DISPLACEMENT_KERNEL_BUFFERS,
        constant CameraUniforms &cam [[buffer(SCR_DISP_BUF_USER0)]],
        constant DofParams      &dof [[buffer(SCR_DISP_BUF_USER1)]],
        uint id [[thread_position_in_grid]])
    {
        RasterBatch batch; uint pointIndex, localOffset;
        if (!scr_resolveDisplacementThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        const float3 p = scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels);
        const float viewDepth = -(cam.modelView * float4(p, 1.0)).z;
        const float band    = cam.focalDistance * dof.band;
        const float falloff = max(cam.focalDistance * dof.falloff, 1e-3);
        const float coc = saturate((abs(viewDepth - cam.focalDistance) - band) / falloff);
        const float3 dir = dofHash3(pointIndex, 0u);
        displacements[pointIndex] = dir * (coc * coc * cam.focalDistance * dof.scatter);
    }
    """

    static let dofTranslucencyKernel = dofStructs + """

    kernel void computeTint(
        SCR_TINT_KERNEL_BUFFERS,
        constant CameraUniforms &cam [[buffer(SCR_TINT_BUF_USER0)]],
        constant DofParams      &dof [[buffer(SCR_TINT_BUF_USER1)]],
        uint id [[thread_position_in_grid]])
    {
        RasterBatch batch; uint pointIndex, localOffset;
        if (!scr_resolveTintThread(id, _scrInfo, batches, batch, pointIndex, localOffset)) return;
        const float3 p = scr_decodePointAt(pointIndex, batch, xyzLow, xyzMed, xyzHigh, levels);
        const float viewDepth = -(cam.modelView * float4(p, 1.0)).z;
        const float band    = cam.focalDistance * dof.band;
        const float falloff = max(cam.focalDistance * dof.falloff, 1e-3);
        const float coc = saturate((abs(viewDepth - cam.focalDistance) - band) / falloff);
        // alpha = coc → rasteriser composites with coverage = 1 - coc (translucent).
        tints[pointIndex] = float4(0.0, 0.0, 0.0, coc * saturate(dof.maxDefocus));
    }
    """
}
