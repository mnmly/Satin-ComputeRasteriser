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

    public static func pack(
        positions: [SIMD3<Float>],
        colors: [SIMD4<Float>],
        pointsPerBatch: Int = computeRasteriserThreadsPerGroup * 80
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

        var batches: [RasterBatch] = []
        var xyzLow = Array(repeating: UInt32(0), count: positions.count)
        var xyzMed = Array(repeating: UInt32(0), count: positions.count)
        var xyzHigh = Array(repeating: UInt32(0), count: positions.count)

        var first = 0
        while first < positions.count {
            let end = min(first + max(pointsPerBatch, 1), positions.count)
            let slice = positions[first ..< end]
            var batchMin = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
            var batchMax = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)

            for position in slice {
                batchMin = simd_min(batchMin, position)
                batchMax = simd_max(batchMax, position)
            }

            let size = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))

            for pointIndex in first ..< end {
                let normalized = simd_clamp((positions[pointIndex] - batchMin) / size, .zero, SIMD3<Float>(repeating: 0.99999994))
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

        let packedColors = colors.map { color -> UInt32 in
            let r = UInt32(simd_clamp(color.x, 0.0, 1.0) * 255.0)
            let g = UInt32(simd_clamp(color.y, 0.0, 1.0) * 255.0)
            let b = UInt32(simd_clamp(color.z, 0.0, 1.0) * 255.0)
            return r | (g << 8) | (b << 16)
        }

        return PackedPointCloud(
            batches: batches,
            files: [RasterFile()],
            xyzLow: xyzLow,
            xyzMed: xyzMed,
            xyzHigh: xyzHigh,
            colors: packedColors,
            boundsMin: boundsMin,
            boundsMax: boundsMax
        )
    }
}

