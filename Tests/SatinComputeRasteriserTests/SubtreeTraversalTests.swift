#if canImport(Metal)
import Foundation
import Metal
import Satin
import Testing
@testable import SatinComputeRasteriser

// `pointClouds` must find clouds anywhere in the rasteriser's subtree, not just
// direct children — so callers can organise clouds under intermediate group
// `Object`s (e.g. a per-capture / per-month parent node) and still have them
// culled, packed, and drawn.
@Test func pointCloudsTraversesNestedGroups() throws {
    let device = try #require(MTLCreateSystemDefaultDevice(), "test requires a Metal device")
    let context = Context(device: device, sampleCount: 1, colorPixelFormat: .rgba8Unorm)
    let ras = ComputeRasteriser(context: context, label: "TraversalTest")
    let cap = ComputeRasteriserCapacity(maxResidentBatches: 4, pointsPerBatch: 8)

    // Direct child (the classic case).
    let direct = ComputeRasteriserPointCloud(context: context, capacity: cap)
    ras.addPointCloud(direct)

    // One level deep, under a group object.
    let group = Object(context: context, label: "2025-05")
    ras.add(group)
    let nestedA = ComputeRasteriserPointCloud(context: context, capacity: cap)
    group.add(nestedA)

    // Two levels deep, under a nested group.
    let subgroup = Object(context: context, label: "sub")
    group.add(subgroup)
    let nestedB = ComputeRasteriserPointCloud(context: context, capacity: cap)
    subgroup.add(nestedB)

    let found = ras.pointClouds
    #expect(found.count == 3)
    #expect(found.contains { $0 === direct })
    #expect(found.contains { $0 === nestedA })
    #expect(found.contains { $0 === nestedB })
}
#endif
