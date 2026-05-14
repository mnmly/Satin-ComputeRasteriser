import Combine
import Metal
import Satin
import SatinComputeRasteriser
import simd
#if canImport(SwiftPDAL)
import SwiftPDAL
#endif

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
    @Published public var enableLODDither: Bool = true
    /// Streaming-mode telemetry. Updated by the StreamingAdapter each frame.
    /// Zero in non-streaming (PLY/fixture) mode.
    @Published public var streamingChunks: Int = 0
    @Published public var streamingPoints: Int = 0
    /// Streaming budget in MB; the adapter passes this to the source's
    /// residency decider to cap working-set bytes.
    @Published public var streamingBudgetMB: Int = 1024
    @Published public var isStreaming: Bool = false

    public init() {}
}

open class ComputeRasteriserAppRenderer: MetalViewRenderer, @unchecked Sendable {
    public lazy var renderer = Renderer(context: defaultContext)
    public lazy var rasteriser = ComputeRasteriser(context: defaultContext)
    public private(set) lazy var pointCloud = ComputeRasteriserPointCloud(
        context: defaultContext,
        packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 28)
    )
    public lazy var scene = Object(context: defaultContext, label: "Compute Rasteriser App", [rasteriser])

    private var currentViewport: SIMD2<Float> = SIMD2(800, 600)

    #if canImport(SwiftPDAL)
    private var streamingAdapter: StreamingAdapter?
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

    public init(
        initialPLYURL: URL? = nil,
        initialMode: ComputeRasteriserMode = .highQualityAverage,
        appState: ComputeRasteriserAppState = ComputeRasteriserAppState()
    ) {
        self.initialPLYURL = initialPLYURL
        self.appState = appState
        self.appState.mode = initialMode
        super.init()
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
        rasteriser.configuration.enableLODDither = appState.enableLODDither
        camera.lookAt(target: .zero)
        cameraController = PerspectiveCameraController(camera: camera, view: metalView)
        cameraController?.defaultDistance = 2.4
        cameraController?.enable()

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
                let opts = StreamingOptions(
                    maxInFlightLoads: cores * 2,
                    decodeConcurrency: cores,
                    driverTickInterval: .milliseconds(16)
                )
                let source = try await SwiftPDAL.CopcStreamingPointCloudSource.open(url, options: opts)
                source.setBudget(budgetBytes)
                await MainActor.run {
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

        let adapter = StreamingAdapter(source: source, cloud: cloud, pixelScale: currentViewport.y * 0.5)
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
