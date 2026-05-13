import Combine
import Metal
import Satin
import SatinComputeRasteriser
import simd

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

    public init() {}
}

open class ComputeRasteriserAppRenderer: MetalViewRenderer, @unchecked Sendable {
    public lazy var renderer = Renderer(context: defaultContext)
    public lazy var rasteriser = ComputeRasteriser(context: defaultContext)
    public lazy var pointCloud = ComputeRasteriserPointCloud(
        context: defaultContext,
        packed: PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 28)
    )
    public lazy var scene = Object(context: defaultContext, label: "Compute Rasteriser App", [rasteriser])
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
        let center = (packed.boundsMin + packed.boundsMax) * 0.5
        let extent = packed.boundsMax - packed.boundsMin
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
