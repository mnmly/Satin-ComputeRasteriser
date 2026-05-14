// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Satin-ComputeRasteriser",
    platforms: [.macOS(.v15)],
    products: [
        .library(
            name: "SatinComputeRasteriser",
            targets: ["SatinComputeRasteriser"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/mnmly/Satin", branch: "feature/2.0-shader-source-transforms"),
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
        .testTarget(
            name: "SatinComputeRasteriserTests",
            dependencies: ["SatinComputeRasteriser"]
        ),
    ]
)
