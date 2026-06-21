// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Satin-ComputeRasteriser",
    platforms: [.macOS(.v15), .iOS(.v18)],
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
        .package(url: "https://github.com/mnmly/Satin", branch: "feature/2.0-metal-4"),
        .package(url: "https://github.com/mnmly/SwiftPDAL", from: "1.14.0"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.5.0"),
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
