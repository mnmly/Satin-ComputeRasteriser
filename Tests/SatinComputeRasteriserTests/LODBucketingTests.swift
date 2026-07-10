import Foundation
import simd
import Testing
@testable import SatinComputeRasteriser

// Slice-1 LOD bucketing (see tasks/todo.md): CPU pack() stores each batch's
// points level-ascending and records cumulative level counts in
// RasterBatch.padding3...6, so the cull pass can bound the draw loops by the
// survivor prefix cum[Lmax]. Pure-Swift coverage — no Metal device needed.

/// Deterministic pseudo-random cloud (LCG) — multiple batches, multiple levels.
private func jitteredCloud(count: Int = 3000, seed: UInt64 = 0x5EED) -> ([SIMD3<Float>], [SIMD4<Float>]) {
    var state = seed
    func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24)
    }
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    positions.reserveCapacity(count)
    colors.reserveCapacity(count)
    for _ in 0 ..< count {
        positions.append(SIMD3<Float>(next() * 10 - 5, next() * 10 - 5, next() * 10 - 5))
        colors.append(SIMD4<Float>(next(), next(), next(), 1))
    }
    return (positions, colors)
}

private func fixtures() -> [PackedPointCloud] {
    let (positions, colors) = jitteredCloud()
    return [
        // Multi-batch, small batches, 4 LOD levels.
        PackedPointCloudFixtures.pack(
            positions: positions,
            colors: colors,
            pointsPerBatch: 512,
            lodLevels: 4,
            coarseVoxelDivisions: 8
        ),
        // Default packing parameters on the stock grid fixture.
        PackedPointCloudFixtures.cubeGrid(pointsPerAxis: 24),
    ]
}

/// The kernel's prefix bound: `Lmax = clamp(int(floor(lodThreshold + 0.5)), 0, 7)`.
private func lmax(for lodThreshold: Float) -> Int {
    min(max(Int(floorf(lodThreshold + 0.5)), 0), 7)
}

@Test func lodCumulativeCountsRoundTripThroughPaddingWords() {
    var batch = RasterBatch(min: .zero, max: .one, numPoints: 800, firstPoint: 0)
    // Fresh batch is the legacy sentinel: all-zero counts, padding6 == 0.
    #expect(batch.lodCumulativeCounts == [UInt16](repeating: 0, count: 8))
    #expect(batch.padding6 == 0)

    batch.setLODCumulativeCounts([1, 5, 5, 100, 100, 100, 700, 800])
    #expect(batch.lodCumulativeCounts.map(Int.init) == [1, 5, 5, 100, 100, 100, 700, 800])
    #expect(batch.padding6 != 0)
}

@Test func packBucketsBatchesByLevelWithMatchingPrefixCounts() {
    for packed in fixtures() {
        #expect(!packed.batches.isEmpty)
        for (batchIndex, batch) in packed.batches.enumerated() {
            let first = Int(batch.firstPoint)
            let slice = packed.levels[first ..< first + Int(batch.numPoints)]

            let nonDecreasing = zip(slice, slice.dropFirst()).allSatisfy { $0 <= $1 }
            #expect(nonDecreasing, "batch \(batchIndex): levels not stored level-ascending")

            var recount = [Int](repeating: 0, count: 8)
            for level in slice { recount[Int(level) & 7] += 1 }
            for level in 1 ..< 8 { recount[level] += recount[level - 1] }

            let stored = batch.lodCumulativeCounts.map(Int.init)
            #expect(stored == recount, "batch \(batchIndex): stored prefix counts disagree with recount")
            #expect(stored[7] == Int(batch.numPoints), "batch \(batchIndex): cum7 must equal numPoints")
            #expect(batch.padding6 != 0, "batch \(batchIndex): bucketed batch must not carry the legacy sentinel")
        }
    }
}

@Test func packPermutationStaysConsistentAfterBucketing() {
    let (positions, colors) = jitteredCloud(count: 2000, seed: 0xB0CE7)
    let packed = PackedPointCloudFixtures.pack(
        positions: positions,
        colors: colors,
        pointsPerBatch: 512,
        lodLevels: 4,
        coarseVoxelDivisions: 8
    )

    #expect(packed.sourceIndices.count == positions.count)
    #expect(packed.orderedPositions.count == positions.count)
    #expect(Set(packed.sourceIndices).count == positions.count, "sourceIndices must stay a permutation")

    func packColor(_ color: SIMD4<Float>) -> UInt32 {
        let r = UInt32(simd_clamp(color.x, 0.0, 1.0) * 255.0)
        let g = UInt32(simd_clamp(color.y, 0.0, 1.0) * 255.0)
        let b = UInt32(simd_clamp(color.z, 0.0, 1.0) * 255.0)
        let a = UInt32(simd_clamp(color.w, 0.0, 1.0) * 255.0)
        return r | (g << 8) | (b << 16) | (a << 24)
    }

    var positionMismatches = 0
    var colorMismatches = 0
    for packedIndex in 0 ..< packed.pointCount {
        let sourceIndex = Int(packed.sourceIndices[packedIndex])
        if packed.orderedPositions[packedIndex] != positions[sourceIndex] { positionMismatches += 1 }
        if packed.colors[packedIndex] != packColor(colors[sourceIndex]) { colorMismatches += 1 }
    }
    #expect(positionMismatches == 0, "\(positionMismatches) orderedPositions do not round-trip through sourceIndices")
    #expect(colorMismatches == 0, "\(colorMismatches) colors do not follow the pack permutation")
}

@Test func survivorPrefixIsExactUpperBoundOnKeepTest() {
    // The draw kernels keep a point iff dither < lodThreshold - level + 0.5
    // with dither in [0, 1), which can only pass when level < lodThreshold + 0.5.
    // Every point outside the cum[Lmax] prefix must fail that necessary
    // condition for any dither value.
    let thresholds: [Float] = [0, 0.4, 0.5, 1.0, 1.7, 3.0, 99]
    for packed in fixtures() {
        for lodThreshold in thresholds {
            let prefixLevel = lmax(for: lodThreshold)
            for (batchIndex, batch) in packed.batches.enumerated() {
                let cumulative = batch.lodCumulativeCounts.map(Int.init)
                let activePoints = cumulative[prefixLevel]
                let first = Int(batch.firstPoint)

                var possibleSurvivorsOutsidePrefix = 0
                for localIndex in activePoints ..< Int(batch.numPoints) {
                    let level = Float(packed.levels[first + localIndex] & 7)
                    if level < lodThreshold + 0.5 { possibleSurvivorsOutsidePrefix += 1 }
                }
                #expect(
                    possibleSurvivorsOutsidePrefix == 0,
                    "threshold \(lodThreshold), batch \(batchIndex): \(possibleSurvivorsOutsidePrefix) points past cum[\(prefixLevel)]=\(activePoints) could still pass the keep-test"
                )
            }
        }
    }
}

@Test func clodOffSentinelThresholdActivatesFullBatch() {
    // CLOD off writes lodThreshold = 99 → Lmax = 7 → activePoints = cum7 ==
    // numPoints, so bucketed batches still draw every resident point.
    #expect(lmax(for: 99) == 7)
    for packed in fixtures() {
        for batch in packed.batches {
            #expect(Int(batch.lodCumulativeCounts[7]) == Int(batch.numPoints))
        }
    }
}
