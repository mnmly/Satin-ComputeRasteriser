import SatinComputeRasteriser
import SwiftUI

@main
struct SatinComputeRasteriserApp: App {
    private let renderer: ComputeRasteriserAppRenderer

    init() {
        renderer = ComputeRasteriserAppRenderer(
            initialPLYURL: Self.plyURL(from: CommandLine.arguments),
            initialMode: Self.mode(from: CommandLine.arguments)
        )
    }

    var body: some Scene {
        WindowGroup("Satin Compute Rasteriser") {
            ComputeRasteriserAppView(renderer: renderer)
                #if os(macOS)
                .frame(minWidth: 640, minHeight: 640)
                #endif
        }
        #if os(macOS)
        .windowResizability(.contentSize)
        #endif
    }

    private static func plyURL(from arguments: [String]) -> URL? {
        guard let index = arguments.firstIndex(of: "--ply"),
              arguments.indices.contains(index + 1) else {
            return nil
        }
        let path = arguments[index + 1]
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(path)
    }

    private static func mode(from arguments: [String]) -> ComputeRasteriserMode {
        guard let index = arguments.firstIndex(of: "--mode"),
              arguments.indices.contains(index + 1) else {
            return .highQualityAverage
        }

        switch arguments[index + 1].lowercased() {
        case "nearest", "nearest-point", "nearestpoint":
            return .nearestPoint
        default:
            return .highQualityAverage
        }
    }
}
