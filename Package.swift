// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Satin-ComputeRasteriser",
    platforms: [.macOS(.v15), .iOS(.v17)],
    products: [
        .library(
            name: "SatinComputeRasteriser",
            targets: ["SatinComputeRasteriser"]
        ),
        .library(
            name: "SatinComputeRasteriserStreaming",
            targets: ["SatinComputeRasteriserStreaming"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/Fabric-Project/Satin", branch: "feature/2.0"),
        // PHASE 3 (revert to from: "1.7.1" / a 1.14.x tag): branch-pin the raw-
        // points emission mode while validating the GPU-pack streaming path.
        .package(url: "https://github.com/mnmly/SwiftPDAL", branch: "raw-points-emission"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "SatinComputeRasteriser",
            dependencies: [
                .product(name: "Satin", package: "Satin"),
            ],
            resources: [
                .copy("Pipelines"),
            ]
        ),
        .target(
            name: "SatinComputeRasteriserStreaming",
            dependencies: [
                "SatinComputeRasteriser",
                .product(name: "Satin", package: "Satin"),
                .product(name: "SwiftPDAL", package: "SwiftPDAL"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "SatinComputeRasteriserTests",
            dependencies: ["SatinComputeRasteriser"]
        ),
    ]
)
