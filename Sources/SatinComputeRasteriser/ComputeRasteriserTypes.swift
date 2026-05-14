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
    case screenSpace = 0
    /// Perspective projection of a world-space sphere radius.
    /// `pointSizeScale` is interpreted as the sphere radius in scene units; FOV and screen height affect size.
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
        applyDisplacement: Bool = false
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
    public var padding: simd_float4x4

    public init(
        transform: simd_float4x4 = matrix_identity_float4x4,
        transformFrustum: simd_float4x4 = matrix_identity_float4x4,
        world: simd_float4x4 = matrix_identity_float4x4
    ) {
        self.transform = transform
        self.transformFrustum = transformFrustum
        self.world = world
        self.padding = matrix_identity_float4x4
    }
}

public struct RasterPixel: Sendable {
    public var depth: UInt32
    public var red: UInt32
    public var green: UInt32
    public var blue: UInt32
    public var count: UInt32
    public var padding: SIMD3<UInt32>

    public init(depth: UInt32 = 0, red: UInt32 = 0, green: UInt32 = 0, blue: UInt32 = 0, count: UInt32 = 0) {
        self.depth = depth
        self.red = red
        self.green = green
        self.blue = blue
        self.count = count
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
        orderedPositions: [SIMD3<Float>] = []
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
