import Foundation
import simd

public let computeRasteriserThreadsPerGroup = 128
public let computeRasteriserSteps30Bit: UInt32 = 1_073_741_824
public let computeRasteriserMask10Bit: UInt32 = 1_023

public enum ComputeRasteriserMode: CaseIterable, Hashable, Sendable {
    case highQualityAverage
    case nearestPoint
}

public enum PointSizeMode: Int32, CaseIterable, Hashable, Sendable {
    /// `pointSizeScale / length(viewSpacePosition)` clamped to [min, max] pixels.
    /// `pointSizeScale` behaves as "pixels at one unit of view distance"; FOV does not affect size.
    /// Under an orthographic projection there is no depth falloff — size is a constant
    /// `pointSizeScale` pixels regardless of distance from the camera.
    case screenSpace = 0
    /// Perspective projection of a world-space sphere radius.
    /// `pointSizeScale` is interpreted as the sphere radius in scene units; FOV and screen height affect size.
    /// Under an orthographic projection there is no depth falloff — `pointSizeScale` maps to a
    /// true, depth-independent world-space size via the projection's fixed pixels-per-world-unit scale.
    case worldSpace = 1
}

public struct ComputeRasteriserConfiguration: Sendable {
    public var mode: ComputeRasteriserMode
    public var depthTolerance: Float
    public var backgroundColor: SIMD4<Float>
    public var enableFrustumCulling: Bool
    public var lodBias: Int
    /// Continuous LOD master switch. When `false`, the cull pass writes a
    /// sentinel `lodThreshold` so the depth/color/nearest passes draw every
    /// resident point regardless of its precomputed level — useful for
    /// comparing the streamed-cloud look against an "all points always"
    /// renderer. ``lodBias`` and ``enableLODDither`` are ignored when off.
    public var enableCLOD: Bool
    public var enableLODDither: Bool
    public var holeFillIterations: Int
    public var colorizeChunks: Bool
    public var colorizeOverdraw: Bool
    public var pointSizeMode: PointSizeMode
    public var minimumPointSize: Float
    public var maximumPointSize: Float
    public var pointSizeScale: Float
    /// If true, the depth/color passes add per-point `displacementBuffer[i]`
    /// (in cloud pack-order) to the decoded position before projection.
    /// Only honored in `.highQualityAverage` mode for now.
    public var applyDisplacement: Bool

    /// If true, the color pass mixes per-point `tintBuffer[i].rgb` into the
    /// stored color weighted by `tintBuffer[i].a`. `tint.a==0` is a no-op,
    /// `tint.a==1` is a full color replacement. Only honored in
    /// `.highQualityAverage` mode for now.
    public var applyTint: Bool

    /// Translucent-defocus mode (weighted-blended order-independent transparency).
    ///
    /// If true (with ``applyTint``), the per-point tint alpha is read as a
    /// circle-of-confusion (0 = in focus, 1 = fully defocused) rather than a
    /// colour-mix weight: defocused points keep their native colour but skip the
    /// depth write and accumulate **coverage-weighted** with NO binary depth test,
    /// so focus→defocus is a smooth blend (no hard occlusion edge) and the sharp
    /// points behind a blurred one show through. A point with no tint pass reads
    /// the zeroed stand-in (alpha 0 → fully opaque), so non-defocused clouds are
    /// unaffected. Resolve: colour = Σ(c·α)/Σα, alpha = 1 − e^(−Σα). Only honored
    /// in `.highQualityAverage` mode. Opt in per-pass via ``TintPass/alphaIsCoverage``.
    ///
    /// **Performance:** dropping the front-surface depth-test early-out means the
    /// colour pass accumulates EVERY point along each ray (all depth layers), not
    /// just the visible surface — cost scales with view-dependent **overdraw** and
    /// is dominated by atomic contention on the pixel buffer. Cheap on thin/flat
    /// clouds, several× heavier on dense/deep ones; throttle with the streaming LOD
    /// budget and point size. (A screen-space post-process DoF would instead be
    /// O(pixels), independent of overdraw.)
    public var tintAlphaIsCoverage: Bool

    /// If true, the final composite writes the cloud's per-pixel reversed-Z
    /// depth into the render pass's depth attachment and depth-tests against it
    /// (`.greaterEqual`). This lets regular Satin meshes (e.g. a selection
    /// bounding box) correctly inter-occlude with the cloud instead of the
    /// cloud always painting on top. Set `false` to restore the legacy
    /// always-on-top overlay (e.g. a composite pass with no depth attachment).
    public var writesSceneDepth: Bool

    /// Motion-blur shutter strength. 0 = off. Scales each point's screen-space
    /// velocity (camera + displacement) into a smear length; the color pass
    /// sweeps the splat across the swept path (coverage/OIT accumulation), so
    /// enabling it also routes the resolve through the coverage path (the cloud
    /// composites as weighted-blended-OIT while blur is on).
    public var motionBlur: Float
    /// Maximum sub-samples swept along the velocity vector (energy split 1/N).
    public var motionBlurSamples: Int
    /// Clamp on the smear length in pixels (caps cost + runaway blur).
    public var motionBlurMaxSpread: Float

    public init(
        mode: ComputeRasteriserMode = .highQualityAverage,
        depthTolerance: Float = 0.01,
        backgroundColor: SIMD4<Float> = SIMD4<Float>(0.0, 0.0, 0.0, 0.0),
        enableFrustumCulling: Bool = true,
        lodBias: Int = 0,
        enableCLOD: Bool = true,
        enableLODDither: Bool = true,
        holeFillIterations: Int = 0,
        colorizeChunks: Bool = false,
        colorizeOverdraw: Bool = false,
        pointSizeMode: PointSizeMode = .screenSpace,
        minimumPointSize: Float = 1.0,
        maximumPointSize: Float = 1.0,
        pointSizeScale: Float = 1.0,
        applyDisplacement: Bool = false,
        applyTint: Bool = false,
        tintAlphaIsCoverage: Bool = false,
        writesSceneDepth: Bool = true,
        motionBlur: Float = 0.0,
        motionBlurSamples: Int = 8,
        motionBlurMaxSpread: Float = 64.0
    ) {
        self.mode = mode
        self.depthTolerance = depthTolerance
        self.backgroundColor = backgroundColor
        self.enableFrustumCulling = enableFrustumCulling
        self.lodBias = lodBias
        self.enableCLOD = enableCLOD
        self.enableLODDither = enableLODDither
        self.holeFillIterations = holeFillIterations
        self.colorizeChunks = colorizeChunks
        self.colorizeOverdraw = colorizeOverdraw
        self.pointSizeMode = pointSizeMode
        self.minimumPointSize = minimumPointSize
        self.maximumPointSize = maximumPointSize
        self.pointSizeScale = pointSizeScale
        self.applyDisplacement = applyDisplacement
        self.applyTint = applyTint
        self.tintAlphaIsCoverage = tintAlphaIsCoverage
        self.writesSceneDepth = writesSceneDepth
        self.motionBlur = motionBlur
        self.motionBlurSamples = motionBlurSamples
        self.motionBlurMaxSpread = motionBlurMaxSpread
    }
}

public struct RasterBatch: Sendable {
    public var state: Int32
    public var minX: Float
    public var minY: Float
    public var minZ: Float
    public var maxX: Float
    public var maxY: Float
    public var maxZ: Float
    public var numPoints: UInt32
    public var firstPoint: UInt32
    public var fileIndex: UInt32
    public var padding3: UInt32
    public var padding4: UInt32
    public var padding5: UInt32
    public var padding6: UInt32
    public var padding7: UInt32
    public var padding8: UInt32

    public init(
        min: SIMD3<Float>,
        max: SIMD3<Float>,
        numPoints: UInt32,
        firstPoint: UInt32,
        fileIndex: UInt32 = 0,
        state: Int32 = 1
    ) {
        self.state = state
        self.minX = min.x
        self.minY = min.y
        self.minZ = min.z
        self.maxX = max.x
        self.maxY = max.y
        self.maxZ = max.z
        self.numPoints = numPoints
        self.firstPoint = firstPoint
        self.fileIndex = fileIndex
        self.padding3 = 0
        self.padding4 = 0
        self.padding5 = 0
        self.padding6 = 0
        self.padding7 = 0
        self.padding8 = 0
    }
}

public struct RasterFile: Sendable {
    public var transform: simd_float4x4
    public var transformFrustum: simd_float4x4
    public var world: simd_float4x4
    /// Previous frame's `transform` (view-projection · world), filled by
    /// `updateFiles`. Used by the color pass to compute per-point screen-space
    /// velocity for motion blur. (Formerly reserved `padding`.)
    public var prevTransform: simd_float4x4

    public init(
        transform: simd_float4x4 = matrix_identity_float4x4,
        transformFrustum: simd_float4x4 = matrix_identity_float4x4,
        world: simd_float4x4 = matrix_identity_float4x4
    ) {
        self.transform = transform
        self.transformFrustum = transformFrustum
        self.world = world
        self.prevTransform = transform
    }
}

public struct RasterPixel: Sendable {
    public var depth: UInt32
    public var red: UInt32
    public var green: UInt32
    public var blue: UInt32
    public var count: UInt32
    /// Σ(coverage·255) accumulated when translucent-defocus is on; 0 otherwise.
    /// Carved from the former 3-word padding, so the stride is unchanged.
    public var weight: UInt32
    public var padding: SIMD2<UInt32>

    public init(depth: UInt32 = 0, red: UInt32 = 0, green: UInt32 = 0, blue: UInt32 = 0, count: UInt32 = 0, weight: UInt32 = 0) {
        self.depth = depth
        self.red = red
        self.green = green
        self.blue = blue
        self.count = count
        self.weight = weight
        self.padding = .zero
    }
}

public struct PackedPointCloud: Sendable {
    public var batches: [RasterBatch]
    public var files: [RasterFile]
    public var xyzLow: [UInt32]
    public var xyzMed: [UInt32]
    public var xyzHigh: [UInt32]
    public var colors: [UInt32]
    public var levels: [UInt8]
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>
    /// Original `[SIMD3<Float>]` reordered into pack order (Morton-sorted).
    /// Empty when a loader didn't preserve them (e.g. PLY paged loaders).
    /// Use this to populate a per-point displacement buffer keyed by the same
    /// `pointIndex` the rasteriser shader uses.
    public var orderedPositions: [SIMD3<Float>]
    /// The pack permutation: `sourceIndices[packedIndex]` is the index of that
    /// point in the loader's **original** (pre-Morton-sort) input arrays. Lets a
    /// caller map a rasteriser `pointIndex` (e.g. from a nearest-point pick) back
    /// to the source point so it can resolve that point's full attribute set.
    /// Empty when a loader didn't build the cloud through ``pack`` (e.g. paged /
    /// streaming loaders that pack on the GPU).
    public var sourceIndices: [UInt32]

    public var pointCount: Int { colors.count }
    public var batchCount: Int { batches.count }

    public init(
        batches: [RasterBatch],
        files: [RasterFile],
        xyzLow: [UInt32],
        xyzMed: [UInt32],
        xyzHigh: [UInt32],
        colors: [UInt32],
        levels: [UInt8],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>,
        orderedPositions: [SIMD3<Float>] = [],
        sourceIndices: [UInt32] = []
    ) {
        self.batches = batches
        self.files = files
        self.xyzLow = xyzLow
        self.xyzMed = xyzMed
        self.xyzHigh = xyzHigh
        self.colors = colors
        self.levels = levels
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.orderedPositions = orderedPositions
        self.sourceIndices = sourceIndices
    }
}

public struct VisibleBatch: Sendable {
    public var batchIndex: UInt32
    public var level: Int32
    public var lodThreshold: Float
    public var padding: UInt32

    public init(batchIndex: UInt32 = 0, level: Int32 = 0, lodThreshold: Float = 0) {
        self.batchIndex = batchIndex
        self.level = level
        self.lodThreshold = lodThreshold
        self.padding = 0
    }
}

public struct CRDispatchArgs: Sendable {
    public var threadgroupsX: UInt32
    public var threadgroupsY: UInt32
    public var threadgroupsZ: UInt32

    public init(threadgroupsX: UInt32 = 0, threadgroupsY: UInt32 = 1, threadgroupsZ: UInt32 = 1) {
        self.threadgroupsX = threadgroupsX
        self.threadgroupsY = threadgroupsY
        self.threadgroupsZ = threadgroupsZ
    }
}

public enum ComputeRasteriserLayout {
    public static let rasterBatchStride = MemoryLayout<RasterBatch>.stride
    public static let rasterFileStride = MemoryLayout<RasterFile>.stride
    public static let rasterPixelStride = MemoryLayout<RasterPixel>.stride
    public static let visibleBatchStride = MemoryLayout<VisibleBatch>.stride
    public static let dispatchArgsStride = MemoryLayout<CRDispatchArgs>.stride
}
