import Foundation
import simd

public enum PackedPointCloudFixtures {
    public static func cubeGrid(pointsPerAxis: Int = 24) -> PackedPointCloud {
        let count = max(pointsPerAxis, 2)
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD4<Float>] = []
        positions.reserveCapacity(count * count * count)
        colors.reserveCapacity(count * count * count)

        for z in 0 ..< count {
            for y in 0 ..< count {
                for x in 0 ..< count {
                    let fx = Float(x) / Float(count - 1)
                    let fy = Float(y) / Float(count - 1)
                    let fz = Float(z) / Float(count - 1)
                    positions.append(SIMD3<Float>(fx - 0.5, fy - 0.5, fz - 0.5))
                    colors.append(SIMD4<Float>(fx, fy, fz, 1.0))
                }
            }
        }

        return pack(positions: positions, colors: colors)
    }

    public static let defaultLODLevels: Int = 4
    public static let defaultCoarseVoxelDivisions: Int = 64

    public static func pack(
        positions: [SIMD3<Float>],
        colors: [SIMD4<Float>],
        pointsPerBatch: Int = computeRasteriserThreadsPerGroup * 80,
        lodLevels: Int = defaultLODLevels,
        coarseVoxelDivisions: Int = defaultCoarseVoxelDivisions
    ) -> PackedPointCloud {
        precondition(positions.count == colors.count, "positions and colors must have the same count")
        guard !positions.isEmpty else {
            return PackedPointCloud(
                batches: [],
                files: [RasterFile()],
                xyzLow: [],
                xyzMed: [],
                xyzHigh: [],
                colors: [],
                levels: [],
                boundsMin: .zero,
                boundsMax: .zero
            )
        }

        var boundsMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var boundsMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        for position in positions {
            boundsMin = simd_min(boundsMin, position)
            boundsMax = simd_max(boundsMax, position)
        }

        let order = mortonOrder(positions: positions, boundsMin: boundsMin, boundsMax: boundsMax)
        let sortedPositions = order.map { positions[$0] }
        let sortedColorsSrc = order.map { colors[$0] }
        let levels = computeLODLevels(
            positions: sortedPositions,
            boundsMin: boundsMin,
            boundsMax: boundsMax,
            lodLevels: max(1, min(lodLevels, 8)),
            coarseVoxelDivisions: max(1, coarseVoxelDivisions)
        )

        var batches: [RasterBatch] = []
        var xyzLow = Array(repeating: UInt32(0), count: sortedPositions.count)
        var xyzMed = Array(repeating: UInt32(0), count: sortedPositions.count)
        var xyzHigh = Array(repeating: UInt32(0), count: sortedPositions.count)

        var first = 0
        while first < sortedPositions.count {
            let end = min(first + max(pointsPerBatch, 1), sortedPositions.count)
            let slice = sortedPositions[first ..< end]
            var batchMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
            var batchMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)

            for position in slice {
                batchMin = simd_min(batchMin, position)
                batchMax = simd_max(batchMax, position)
            }

            let size = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))

            for pointIndex in first ..< end {
                let normalized = simd_clamp((sortedPositions[pointIndex] - batchMin) / size, .zero, SIMD3<Float>(repeating: 0.99999994))
                let q = SIMD3<UInt32>(
                    UInt32(normalized.x * Float(computeRasteriserSteps30Bit - 1)),
                    UInt32(normalized.y * Float(computeRasteriserSteps30Bit - 1)),
                    UInt32(normalized.z * Float(computeRasteriserSteps30Bit - 1))
                )

                let xLow = (q.x >> 20) & computeRasteriserMask10Bit
                let yLow = (q.y >> 20) & computeRasteriserMask10Bit
                let zLow = (q.z >> 20) & computeRasteriserMask10Bit
                let xMed = (q.x >> 10) & computeRasteriserMask10Bit
                let yMed = (q.y >> 10) & computeRasteriserMask10Bit
                let zMed = (q.z >> 10) & computeRasteriserMask10Bit
                let xHigh = q.x & computeRasteriserMask10Bit
                let yHigh = q.y & computeRasteriserMask10Bit
                let zHigh = q.z & computeRasteriserMask10Bit

                xyzLow[pointIndex] = xLow | (yLow << 10) | (zLow << 20)
                xyzMed[pointIndex] = xMed | (yMed << 10) | (zMed << 20)
                xyzHigh[pointIndex] = xHigh | (yHigh << 10) | (zHigh << 20)
            }

            batches.append(
                RasterBatch(
                    min: batchMin,
                    max: batchMax,
                    numPoints: UInt32(end - first),
                    firstPoint: UInt32(first),
                    fileIndex: 0
                )
            )
            first = end
        }

        let packedColors = sortedColorsSrc.map { color -> UInt32 in
            let r = UInt32(simd_clamp(color.x, 0.0, 1.0) * 255.0)
            let g = UInt32(simd_clamp(color.y, 0.0, 1.0) * 255.0)
            let b = UInt32(simd_clamp(color.z, 0.0, 1.0) * 255.0)
            let a = UInt32(simd_clamp(color.w, 0.0, 1.0) * 255.0)
            return r | (g << 8) | (b << 16) | (a << 24)
        }

        return PackedPointCloud(
            batches: batches,
            files: [RasterFile()],
            xyzLow: xyzLow,
            xyzMed: xyzMed,
            xyzHigh: xyzHigh,
            colors: packedColors,
            levels: levels,
            boundsMin: boundsMin,
            boundsMax: boundsMax
        )
    }
}

// Z-order points so consecutive entries are spatially close. Tighter per-batch
// AABBs (better frustum culling and precision selection) and better cache
// coherency for neighbouring threadgroup reads in the rasteriser passes.
private func mortonOrder(
    positions: [SIMD3<Float>],
    boundsMin: SIMD3<Float>,
    boundsMax: SIMD3<Float>
) -> [Int] {
    let count = positions.count
    let extent = simd_max(boundsMax - boundsMin, SIMD3<Float>(repeating: 0.000001))
    let scale = SIMD3<Float>(repeating: 1023.0) / extent

    var keys = [UInt32](repeating: 0, count: count)
    for i in 0 ..< count {
        let normalized = simd_clamp((positions[i] - boundsMin) * scale, .zero, SIMD3<Float>(repeating: 1023.0))
        let qx = UInt32(normalized.x)
        let qy = UInt32(normalized.y)
        let qz = UInt32(normalized.z)
        keys[i] = (mortonSpread10(qx) << 2) | (mortonSpread10(qy) << 1) | mortonSpread10(qz)
    }

    var indices = Array(0 ..< count)
    indices.sort { keys[$0] < keys[$1] }
    return indices
}

// Density-aware LOD: for each level (coarsest first) assign points that
// occupy a previously-empty voxel. Coarsest level gets the most spatially
// representative subset; finest level catches the remainder. Levels stored
// in top 3 bits of the packed color word (0 = coarsest visible at distance).
private func computeLODLevels(
    positions: [SIMD3<Float>],
    boundsMin: SIMD3<Float>,
    boundsMax: SIMD3<Float>,
    lodLevels: Int,
    coarseVoxelDivisions: Int
) -> [UInt8] {
    let count = positions.count
    let maxLevel = UInt8(lodLevels - 1)
    var levels = [UInt8](repeating: maxLevel, count: count)
    guard lodLevels > 1 else { return levels }

    let extent = simd_max(boundsMax - boundsMin, SIMD3<Float>(repeating: 0.000001))
    let longestAxis = max(extent.x, max(extent.y, extent.z))
    let baseVoxel = longestAxis / Float(coarseVoxelDivisions)

    var occupied: Set<SIMD3<Int32>> = []
    for level in 0 ..< (lodLevels - 1) {
        let voxelSize = baseVoxel * powf(0.5, Float(level))
        let invVoxel = 1.0 / max(voxelSize, 0.000001)
        occupied.removeAll(keepingCapacity: true)
        for i in 0 ..< count {
            if levels[i] != maxLevel { continue }
            let local = (positions[i] - boundsMin) * invVoxel
            let cell = SIMD3<Int32>(Int32(local.x.rounded(.down)), Int32(local.y.rounded(.down)), Int32(local.z.rounded(.down)))
            if occupied.insert(cell).inserted {
                levels[i] = UInt8(level)
            }
        }
    }
    return levels
}

private func mortonSpread10(_ value: UInt32) -> UInt32 {
    var x = value & 0x3ff
    x = (x | (x << 16)) & 0x030000ff
    x = (x | (x << 8))  & 0x0300f00f
    x = (x | (x << 4))  & 0x030c30c3
    x = (x | (x << 2))  & 0x09249249
    return x
}

