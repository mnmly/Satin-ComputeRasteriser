import Foundation
import simd

struct NearestPointCPUReferenceResult: Sendable {
    var depths: [UInt32]
    var indices: [UInt32]

    func colorBuffer(from colors: [UInt32], background: UInt32 = 0) -> [UInt32] {
        indices.map { index in
            index == UInt32.max ? background : colors[Int(index)]
        }
    }
}

enum NearestPointCPUReference {
    static func render(
        packed: PackedPointCloud,
        width: Int,
        height: Int,
        viewMatrix: simd_float4x4 = matrix_identity_float4x4,
        projectionMatrix: simd_float4x4 = matrix_identity_float4x4,
        enableFrustumCulling: Bool = false
    ) -> NearestPointCPUReferenceResult {
        let pixelCount = max(width, 0) * max(height, 0)
        var depths = Array(repeating: UInt32(0), count: pixelCount)
        var indices = Array(repeating: UInt32.max, count: pixelCount)
        guard width > 0, height > 0 else {
            return NearestPointCPUReferenceResult(depths: depths, indices: indices)
        }

        for batch in packed.batches {
            let file = packed.files[Int(batch.fileIndex)]
            let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ)
            let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ)
            if enableFrustumCulling, !intersectsFrustum(file.transformFrustum, batchMin: batchMin, batchMax: batchMax) {
                continue
            }

            let level = precisionLevel(
                batch: batch,
                file: file,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                imageSize: SIMD2<Int32>(Int32(width), Int32(height))
            )

            for localIndex in 0 ..< Int(batch.numPoints) {
                let pointIndex = Int(batch.firstPoint) + localIndex
                let point = decodePoint(pointIndex: pointIndex, batch: batch, packed: packed, level: level)
                let clip = simd_mul(file.transform, SIMD4<Float>(point, 1.0))
                if clip.w <= 0.0 {
                    continue
                }

                let ndc = SIMD3<Float>(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
                if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z <= 0.0 || ndc.z > 1.0 {
                    continue
                }

                var pixelCoord = SIMD2<Int32>(
                    Int32((ndc.x * 0.5 + 0.5) * Float(width)),
                    Int32((ndc.y * 0.5 + 0.5) * Float(height))
                )
                if pixelCoord.x < 0 || pixelCoord.x >= width || pixelCoord.y < 0 || pixelCoord.y >= height {
                    continue
                }
                pixelCoord.y = Int32(height) - 1 - pixelCoord.y

                let pixelIndex = Int(pixelCoord.y) * width + Int(pixelCoord.x)
                let depth = depthToUIntReverseZ(ndc.z)
                if depth > depths[pixelIndex] {
                    depths[pixelIndex] = depth
                    indices[pixelIndex] = UInt32(pointIndex)
                } else if depth == depths[pixelIndex] {
                    indices[pixelIndex] = min(indices[pixelIndex], UInt32(pointIndex))
                }
            }
        }

        return NearestPointCPUReferenceResult(depths: depths, indices: indices)
    }

    private static func depthToUIntReverseZ(_ ndcZ: Float) -> UInt32 {
        UInt32(min(max(ndcZ, 0.0), 1.0) * Float(UInt32.max - 1))
    }

    private static func decodePoint(pointIndex: Int, batch: RasterBatch, packed: PackedPointCloud, level: Int) -> SIMD3<Float> {
        let batchMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ)
        let batchMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ)
        let batchSize = max(batchMax - batchMin, SIMD3<Float>(repeating: 0.000001))

        if level == 0 {
            let low = packed.xyzLow[pointIndex]
            let med = packed.xyzMed[pointIndex]
            let high = packed.xyzHigh[pointIndex]
            let x = (unpack10(low, shift: 0) << 20) | (unpack10(med, shift: 0) << 10) | unpack10(high, shift: 0)
            let y = (unpack10(low, shift: 10) << 20) | (unpack10(med, shift: 10) << 10) | unpack10(high, shift: 10)
            let z = (unpack10(low, shift: 20) << 20) | (unpack10(med, shift: 20) << 10) | unpack10(high, shift: 20)
            return SIMD3<Float>(Float(x), Float(y), Float(z)) * (batchSize / Float(computeRasteriserSteps30Bit)) + batchMin
        }

        if level == 1 {
            let low = packed.xyzLow[pointIndex]
            let med = packed.xyzMed[pointIndex]
            let x = (unpack10(low, shift: 0) << 10) | unpack10(med, shift: 0)
            let y = (unpack10(low, shift: 10) << 10) | unpack10(med, shift: 10)
            let z = (unpack10(low, shift: 20) << 10) | unpack10(med, shift: 20)
            return SIMD3<Float>(Float(x), Float(y), Float(z)) * (batchSize / 1_048_576.0) + batchMin
        }

        let low = packed.xyzLow[pointIndex]
        let x = unpack10(low, shift: 0)
        let y = unpack10(low, shift: 10)
        let z = unpack10(low, shift: 20)
        return SIMD3<Float>(Float(x), Float(y), Float(z)) * (batchSize / 1_024.0) + batchMin
    }

    private static func unpack10(_ encoded: UInt32, shift: UInt32) -> UInt32 {
        (encoded >> shift) & computeRasteriserMask10Bit
    }

    private static func precisionLevel(
        batch: RasterBatch,
        file: RasterFile,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        imageSize: SIMD2<Int32>
    ) -> Int {
        let wgMin = SIMD3<Float>(batch.minX, batch.minY, batch.minZ)
        let wgMax = SIMD3<Float>(batch.maxX, batch.maxY, batch.maxZ)
        let wgCenter = (wgMin + wgMax) * 0.5
        let wgRadius = simd_distance(wgMin, wgMax)

        let viewCenter = simd_mul(simd_mul(viewMatrix, file.world), SIMD4<Float>(wgCenter, 1.0))
        let viewEdge = viewCenter + SIMD4<Float>(wgRadius, 0.0, 0.0, 0.0)
        var projCenter = simd_mul(projectionMatrix, viewCenter)
        var projEdge = simd_mul(projectionMatrix, viewEdge)

        projCenter.x /= max(abs(projCenter.w), 0.000001)
        projCenter.y /= max(abs(projCenter.w), 0.000001)
        projEdge.x /= max(abs(projEdge.w), 0.000001)
        projEdge.y /= max(abs(projEdge.w), 0.000001)

        let screenCenter = SIMD2<Float>(Float(imageSize.x), Float(imageSize.y)) * (SIMD2<Float>(projCenter.x, projCenter.y) + 1.0) * 0.5
        let screenEdge = SIMD2<Float>(Float(imageSize.x), Float(imageSize.y)) * (SIMD2<Float>(projEdge.x, projEdge.y) + 1.0) * 0.5
        let pixelSize = simd_distance(screenEdge, screenCenter)

        if pixelSize < 100.0 { return 4 }
        if pixelSize < 200.0 { return 3 }
        if pixelSize < 500.0 { return 2 }
        if pixelSize < 10000.0 { return 1 }
        return 0
    }

    private static func intersectsFrustum(_ m: simd_float4x4, batchMin: SIMD3<Float>, batchMax: SIMD3<Float>) -> Bool {
        let planes = [
            makePlane(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]),
            makePlane(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]),
            makePlane(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]),
            makePlane(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]),
            makePlane(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]),
            makePlane(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]),
        ]

        for plane in planes {
            let p = SIMD3<Float>(
                plane.normal.x > 0.0 ? batchMax.x : batchMin.x,
                plane.normal.y > 0.0 ? batchMax.y : batchMin.y,
                plane.normal.z > 0.0 ? batchMax.z : batchMin.z
            )
            if simd_dot(plane.normal, p) + plane.offset < 0.0 {
                return false
            }
        }
        return true
    }

    private static func makePlane(_ x: Float, _ y: Float, _ z: Float, _ w: Float) -> (normal: SIMD3<Float>, offset: Float) {
        let length = max(simd_length(SIMD3<Float>(x, y, z)), 0.000001)
        return (SIMD3<Float>(x, y, z) / length, w / length)
    }
}
