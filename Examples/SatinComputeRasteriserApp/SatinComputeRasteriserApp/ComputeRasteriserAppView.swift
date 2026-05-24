import Satin
import SatinComputeRasteriser
import SwiftUI
import UniformTypeIdentifiers

public struct ComputeRasteriserAppView: View {
    @ObservedObject private var appState: ComputeRasteriserAppState
    private let renderer: ComputeRasteriserAppRenderer
    @State private var isImporterPresented = false
    @State private var isCOPCImporterPresented = false
    @State private var isSettingsPresented = false

    public init(renderer: ComputeRasteriserAppRenderer) {
        self.renderer = renderer
        self.appState = renderer.appState
    }

    public var body: some View {
        NavigationStack {
            SatinMetalView(renderer: renderer)
                .ignoresSafeArea(edges: .bottom)
                .navigationTitle(appState.status)
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                .toolbarBackground(.thinMaterial, for: .navigationBar)
                .toolbarBackground(.visible, for: .navigationBar)
                #endif
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Menu {
                            Button {
                                isImporterPresented = true
                            } label: {
                                Label("Open PLY…", systemImage: "doc")
                            }
                            #if canImport(SwiftPDAL)
                            Button {
                                isCOPCImporterPresented = true
                            } label: {
                                Label("Open COPC…", systemImage: "globe")
                            }
                            #endif
                        } label: {
                            Label("Open", systemImage: "plus")
                        }
                    }
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            isSettingsPresented = true
                        } label: {
                            Label("Settings", systemImage: "slider.horizontal.3")
                        }
                    }
                }
                .safeAreaInset(edge: .bottom) {
                    if appState.errorMessage != nil || appState.isStreamingTelemetryVisible {
                        statusOverlay
                            .padding(.horizontal)
                            .padding(.bottom, 8)
                    }
                }
        }
        .sheet(isPresented: $isSettingsPresented) {
            SettingsSheet(appState: appState, renderer: renderer)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
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
        #if canImport(SwiftPDAL)
        .fileImporter(
            isPresented: $isCOPCImporterPresented,
            allowedContentTypes: [.copcLAZ],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case let .success(urls):
                if let url = urls.first {
                    renderer.loadCOPC(url: url)
                }
            case let .failure(error):
                appState.errorMessage = error.localizedDescription
            }
        }
        #endif
    }

    @ViewBuilder
    private var statusOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let error = appState.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(3)
            }
            #if canImport(SwiftPDAL)
            if appState.isStreaming {
                Text("\(appState.streamingChunks) chunks · \(appState.streamingPoints) pts")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            #endif
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

private extension ComputeRasteriserAppState {
    var isStreamingTelemetryVisible: Bool {
        #if canImport(SwiftPDAL)
        return isStreaming
        #else
        return false
        #endif
    }
}

private struct SettingsSheet: View {
    @ObservedObject var appState: ComputeRasteriserAppState
    let renderer: ComputeRasteriserAppRenderer
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Render mode") {
                    Picker("Mode", selection: Binding(
                        get: { appState.mode },
                        set: { renderer.setMode($0) }
                    )) {
                        Text("High Quality Average").tag(ComputeRasteriserMode.highQualityAverage)
                        Text("Nearest Point").tag(ComputeRasteriserMode.nearestPoint)
                    }
                    .pickerStyle(.segmented)
                }

                Section("Point sizing") {
                    Picker("Mode", selection: Binding(
                        get: { appState.pointSizeMode },
                        set: { renderer.setPointSizing(mode: $0) }
                    )) {
                        Text("Screen").tag(PointSizeMode.screenSpace)
                        Text("World").tag(PointSizeMode.worldSpace)
                    }
                    .pickerStyle(.segmented)

                    sliderRow(
                        title: "Maximum",
                        value: Binding(
                            get: { Double(appState.maximumPointSize) },
                            set: { renderer.setPointSizing(maximum: Float($0)) }
                        ),
                        range: 1 ... 128,
                        formatter: { String(format: "%.0f", $0) }
                    )

                    sliderRow(
                        title: "Scale",
                        value: Binding(
                            get: { Double(appState.pointSizeScale) },
                            set: { renderer.setPointSizing(scale: Float($0)) }
                        ),
                        range: appState.pointSizeMode == .worldSpace ? 0.001 ... 0.1 : 1 ... 16,
                        formatter: { String(format: appState.pointSizeMode == .worldSpace ? "%.3f" : "%.1f", $0) }
                    )
                }

                Section("LOD & culling") {
                    Stepper(
                        "LOD bias: \(appState.lodBias)",
                        value: Binding(
                            get: { appState.lodBias },
                            set: { renderer.setLODBias($0) }
                        ),
                        in: -3 ... 7
                    )
                    Toggle("Frustum culling", isOn: Binding(
                        get: { appState.enableFrustumCulling },
                        set: { renderer.setFrustumCulling($0) }
                    ))
                    Toggle("Continuous LOD", isOn: Binding(
                        get: { appState.enableCLOD },
                        set: { renderer.setCLOD($0) }
                    ))
                    Toggle("LOD dither", isOn: Binding(
                        get: { appState.enableLODDither },
                        set: { renderer.setLODDither($0) }
                    ))
                    .disabled(!appState.enableCLOD)
                }

                Section("Rendering") {
                    Toggle("Colorize chunks", isOn: Binding(
                        get: { appState.colorizeChunks },
                        set: { renderer.setColorizeChunks($0) }
                    ))
                    Stepper(
                        "Hole fill: \(appState.holeFillIterations)",
                        value: Binding(
                            get: { appState.holeFillIterations },
                            set: { renderer.setHoleFillIterations($0) }
                        ),
                        in: 0 ... 4
                    )
                }

                #if canImport(SwiftPDAL)
                if appState.isStreaming {
                    Section("Streaming") {
                        LabeledContent("Resident") {
                            Text("\(appState.streamingChunks) chunks · \(appState.streamingPoints) pts")
                                .monospacedDigit()
                        }
                        sliderRow(
                            title: "Budget",
                            value: Binding(
                                get: { Double(appState.streamingBudgetMB) },
                                set: { renderer.setStreamingBudget(MB: Int($0)) }
                            ),
                            range: 256 ... 16384,
                            step: 256,
                            formatter: { "\(Int($0)) MB" }
                        )
                        Picker("Residency", selection: Binding(
                            get: { appState.streamingResidency },
                            set: { renderer.setResidency($0) }
                        )) {
                            Text("Halo").tag(StreamingResidencyChoice.halo)
                            Text("Distance").tag(StreamingResidencyChoice.distance)
                        }
                        .pickerStyle(.segmented)
                    }
                }
                #endif
            }
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    @ViewBuilder
    private func sliderRow(
        title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double? = nil,
        formatter: @escaping (Double) -> String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(formatter(value.wrappedValue))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            if let step {
                Slider(value: value, in: range, step: step)
            } else {
                Slider(value: value, in: range)
            }
        }
    }
}

extension UTType {
    static let ply = UTType(filenameExtension: "ply") ?? .data
    /// COPC files use the LAZ extension. The fileImporter shouldn't reject
    /// `.las` or non-COPC `.laz` either — SwiftPDAL's open call surfaces the
    /// "not a COPC" error through the standard error path.
    static let copcLAZ = UTType(filenameExtension: "laz") ?? .data
}
