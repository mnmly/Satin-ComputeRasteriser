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

        var order = mortonOrder(positions: positions, boundsMin: boundsMin, boundsMax: boundsMax)
        var sortedPositions = order.map { positions[$0] }
        var levels = computeLODLevels(
            positions: sortedPositions,
            boundsMin: boundsMin,
            boundsMax: boundsMax,
            lodLevels: max(1, min(lodLevels, 8)),
            coarseVoxelDivisions: max(1, coarseVoxelDivisions)
        )

        // Bucket each batch slice level-ascending (stable, so Morton order is
        // preserved within a level) and compose the permutation into `order`,
        // so colors — gathered below — plus `sourceIndices`/`orderedPositions`
        // all follow the same final point order. The cull pass uses the
        // per-batch cumulative counts to bound the draw loops.
        let batchStride = max(pointsPerBatch, 1)
        precondition(batchStride <= 65535, "pointsPerBatch must fit the uint16 LOD prefix counts")
        bucketSortBatchSlicesByLevel(
            order: &order,
            positions: &sortedPositions,
            levels: &levels,
            batchStride: batchStride
        )

        let sortedColorsSrc = order.map { colors[$0] }

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

            var batch = RasterBatch(
                min: batchMin,
                max: batchMax,
                numPoints: UInt32(end - first),
                firstPoint: UInt32(first),
                fileIndex: 0
            )
            var cumulative = [Int](repeating: 0, count: 8)
            for pointIndex in first ..< end {
                cumulative[Int(levels[pointIndex]) & 7] += 1
            }
            for level in 1 ..< 8 {
                cumulative[level] += cumulative[level - 1]
            }
            batch.setLODCumulativeCounts(cumulative)
            batches.append(batch)
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
            boundsMax: boundsMax,
            orderedPositions: sortedPositions,
            sourceIndices: order.map { UInt32($0) }
        )
    }
}

// Stable per-batch-slice 8-bucket counting sort by LOD level: within each
// [first, first+batchStride) slice, points are reordered level-ascending while
// preserving Morton order inside each level. The identical permutation is
// applied to `order` so every array derived from it stays consistent with the
// permuted positions/levels.
private func bucketSortBatchSlicesByLevel(
    order: inout [Int],
    positions: inout [SIMD3<Float>],
    levels: inout [UInt8],
    batchStride: Int
) {
    let count = positions.count
    var first = 0
    while first < count {
        let end = min(first + batchStride, count)

        var cursors = [Int](repeating: 0, count: 9)
        for i in first ..< end {
            cursors[(Int(levels[i]) & 7) + 1] += 1
        }
        for level in 1 ..< 9 {
            cursors[level] += cursors[level - 1]
        }

        // permutation[j] = source index (in the whole array) of the point
        // that lands at slice-relative position j.
        var permutation = [Int](repeating: 0, count: end - first)
        for i in first ..< end {
            let level = Int(levels[i]) & 7
            permutation[cursors[level]] = i
            cursors[level] += 1
        }

        let orderSlice = permutation.map { order[$0] }
        let positionSlice = permutation.map { positions[$0] }
        let levelSlice = permutation.map { levels[$0] }
        for j in 0 ..< permutation.count {
            order[first + j] = orderSlice[j]
            positions[first + j] = positionSlice[j]
            levels[first + j] = levelSlice[j]
        }

        first = end
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

    // Voxel-occupancy dedup keyed by a single UInt64 (21 bits/axis) rather than
    // SIMD3<Int32> — hashing one integer is far cheaper than Swift's per-
    // component Hasher combine. Cells are non-negative (positions >= boundsMin)
    // and node-local, so the pack is collision-free over the values that occur.
    // Matches SwiftPDAL ChunkPacker.computeLODLevels.
    @inline(__always)
    func cellKey(_ local: SIMD3<Float>) -> UInt64 {
        let cx = UInt64(max(0, Int32(local.x.rounded(.down)))) & 0x1F_FFFF
        let cy = UInt64(max(0, Int32(local.y.rounded(.down)))) & 0x1F_FFFF
        let cz = UInt64(max(0, Int32(local.z.rounded(.down)))) & 0x1F_FFFF
        return (cx << 42) | (cy << 21) | cz
    }

    // Open-addressing hash set of occupied voxel keys, replacing Swift's
    // `Set<UInt64>` — whose SipHash + CoW/uniqueness machinery dominated the
    // streaming-decode CPU profile (~12.5% of total process CPU: Hasher._hash,
    // _Variant.insert, isUniquelyReferenced). One `[UInt64]` table, sized once
    // for the whole call and reused across levels, with Fibonacci hashing and
    // linear probing in an unsafe buffer (no bounds/uniqueness checks in the
    // probe). Mirrors `ChunkPacker.computeLODLevels` in SwiftPDAL — keep the
    // two in sync.
    //
    // Sentinel is `UInt64.max`: cellKey packs 21 bits/axis into bits 0..62, so
    // bit 63 is always 0 and a real key can never equal the sentinel.
    //
    // Capacity = next power of two ≥ 2·count (load factor ≤ 0.5). Sizing by the
    // total count — not by per-level unclaimed counts — is deliberate: the
    // eligible set shrinks with depth but the number of distinct occupied
    // voxels (the actual live entries) grows, and `count` bounds both at every
    // level, so one allocation stays under 0.5 load throughout. Per-level
    // resizing would only shave the O(capacity) reset, which is already
    // dominated by the O(count) scan — not worth the extra bookkeeping.
    let sentinel = UInt64.max
    var capacity = 1
    while capacity < count * 2 { capacity <<= 1 }
    let mask = capacity - 1
    let shift = UInt64(64 - capacity.trailingZeroBitCount)

    var table = [UInt64](repeating: sentinel, count: capacity)
    levels.withUnsafeMutableBufferPointer { lv in
        positions.withUnsafeBufferPointer { pos in
            table.withUnsafeMutableBufferPointer { t in
                for level in 0 ..< (lodLevels - 1) {
                    let voxelSize = baseVoxel * powf(0.5, Float(level))
                    let invVoxel = 1.0 / max(voxelSize, 0.000001)
                    // Level 0 uses the freshly sentinel-filled table; refill for
                    // each subsequent level. No "skip refill when no inserts"
                    // guard: the first eligible point always inserts, so at
                    // these coarse levels the guard would never fire.
                    if level != 0 {
                        for j in 0 ..< capacity { t[j] = sentinel }
                    }
                    for i in 0 ..< count {
                        if lv[i] != maxLevel { continue }
                        let local = (pos[i] - boundsMin) * invVoxel
                        let key = cellKey(local)
                        var slot = Int((key &* 0x9E37_79B9_7F4A_7C15) >> shift)
                        while true {
                            let cur = t[slot]
                            if cur == sentinel {
                                t[slot] = key
                                lv[i] = UInt8(level)
                                break
                            }
                            if cur == key { break }
                            slot = (slot + 1) & mask
                        }
                    }
                }
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

