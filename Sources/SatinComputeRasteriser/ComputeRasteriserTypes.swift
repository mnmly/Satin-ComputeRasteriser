import Foundation
import simd

public let computeRasteriserThreadsPerGroup = 128
public let computeRasteriserSteps30Bit: UInt32 = 1_073_741_824
public let computeRasteriserMask10Bit: UInt32 = 1_023

public enum ComputeRasteriserMode: CaseIterable, Hashable, Sendable {
    case highQualityAverage
    case nearestPoint
}

public struct ComputeRasteriserConfiguration: Sendable {
    public var mode: ComputeRasteriserMode
    public var depthTolerance: Float
    public var backgroundColor: SIMD4<Float>
    public var enableFrustumCulling: Bool
    public var colorizeChunks: Bool
    public var colorizeOverdraw: Bool

    public init(
        mode: ComputeRasteriserMode = .highQualityAverage,
        depthTolerance: Float = 0.01,
        backgroundColor: SIMD4<Float> = SIMD4<Float>(0.0, 0.0, 0.0, 0.0),
        enableFrustumCulling: Bool = true,
        colorizeChunks: Bool = false,
        colorizeOverdraw: Bool = false
    ) {
        self.mode = mode
        self.depthTolerance = depthTolerance
        self.backgroundColor = backgroundColor
        self.enableFrustumCulling = enableFrustumCulling
        self.colorizeChunks = colorizeChunks
        self.colorizeOverdraw = colorizeOverdraw
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
        state: Int32 = 0
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
    public var boundsMin: SIMD3<Float>
    public var boundsMax: SIMD3<Float>

    public var pointCount: Int { colors.count }
    public var batchCount: Int { batches.count }

    public init(
        batches: [RasterBatch],
        files: [RasterFile],
        xyzLow: [UInt32],
        xyzMed: [UInt32],
        xyzHigh: [UInt32],
        colors: [UInt32],
        boundsMin: SIMD3<Float>,
        boundsMax: SIMD3<Float>
    ) {
        self.batches = batches
        self.files = files
        self.xyzLow = xyzLow
        self.xyzMed = xyzMed
        self.xyzHigh = xyzHigh
        self.colors = colors
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
    }
}

public enum ComputeRasteriserLayout {
    public static let rasterBatchStride = MemoryLayout<RasterBatch>.stride
    public static let rasterFileStride = MemoryLayout<RasterFile>.stride
    public static let rasterPixelStride = MemoryLayout<RasterPixel>.stride
}
