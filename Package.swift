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
