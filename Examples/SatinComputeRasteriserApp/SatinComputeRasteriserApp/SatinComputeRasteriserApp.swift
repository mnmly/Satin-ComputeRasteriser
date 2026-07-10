import SatinComputeRasteriser
import SwiftUI

@main
struct SatinComputeRasteriserApp: App {
    private let renderer: ComputeRasteriserAppRenderer

    init() {
        renderer = ComputeRasteriserAppRenderer(
            initialPLYURL: Self.plyURL(from: CommandLine.arguments),
            initialCOPCURLs: Self.copcURLs(from: CommandLine.arguments),
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

    /// Every repeatable `--copc <path>` pair, resolved to a file URL. Paths
    /// that don't exist are skipped (one log line each) so a capture launched
    /// with `--copc a --copc b --copc c` degrades gracefully.
    private static func copcURLs(from arguments: [String]) -> [URL] {
        var urls: [URL] = []
        var index = 0
        while index < arguments.count {
            defer { index += 1 }
            guard arguments[index] == "--copc", arguments.indices.contains(index + 1) else { continue }
            index += 1
            let path = arguments[index]
            let url = path.hasPrefix("/")
                ? URL(fileURLWithPath: path)
                : URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: url.path) {
                urls.append(url)
            } else {
                print("[SatinComputeRasteriserApp] --copc: skipping missing file \(url.path)")
            }
        }
        return urls
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
