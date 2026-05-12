import Satin
import SatinComputeRasteriser
import SwiftUI
import UniformTypeIdentifiers

public struct ComputeRasteriserAppView: View {
    @ObservedObject private var appState: ComputeRasteriserAppState
    private let renderer: ComputeRasteriserAppRenderer
    @State private var isImporterPresented = false

    public init(renderer: ComputeRasteriserAppRenderer) {
        self.renderer = renderer
        self.appState = renderer.appState
    }

    public var body: some View {
        ZStack(alignment: .topLeading) {
            SatinMetalView(renderer: renderer)
            HStack(spacing: 10) {
                Button("Open PLY") {
                    isImporterPresented = true
                }
                Picker("Mode", selection: Binding(
                    get: { appState.mode },
                    set: { renderer.setMode($0) }
                )) {
                    Text("HQS").tag(ComputeRasteriserMode.highQualityAverage)
                    Text("Nearest").tag(ComputeRasteriserMode.nearestPoint)
                }
                .pickerStyle(.segmented)
                .frame(width: 150)
                Text(appState.status)
                    .lineLimit(1)
                if let error = appState.errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
            }
            .padding(10)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(12)
        }
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [.ply],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case let .success(urls):
                if let url = urls.first {
                    renderer.loadPLY(url: url)
                }
            case let .failure(error):
                appState.errorMessage = error.localizedDescription
            }
        }
    }
}

extension UTType {
    static let ply = UTType(filenameExtension: "ply") ?? .data
}
