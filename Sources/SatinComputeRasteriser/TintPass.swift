import Combine
import Foundation
import Metal
import Satin

/// Drives an on-GPU color-tint compute pass for a
/// ``ComputeRasteriserPointCloud``. Mirror of ``DisplacementPass``: owns the
/// pipeline build (with live reload), auto-allocates the cloud's
/// `tintBuffer`, flips `ComputeRasteriserConfiguration.applyTint = true` on
/// first encode, and binds the cloud's batches + xyz + levels + tint
/// buffers each frame. The sketch only writes the kernel + any extra
/// uniforms.
///
/// **Sketch contract.** The user-supplied `.metal` file is concatenated
/// with the package's tint preamble before compile. The preamble exposes:
///
///   * `RasterBatch`, `scr_decodePointAt`, and `scr_resolveTintThread` —
///     all identical to their displacement counterparts (point decode +
///     thread/slot resolution).
///   * `SCR_TINT_KERNEL_BUFFERS` macro — expands to the buffer declarations
///     the pass binds automatically (slots 0–7), including `colors` (the
///     native packed-RGBA8 per-point colour). User uniforms bind at
///     ``TintPass/bufferUser0`` (slot 8) onward.
///   * `scr_decodeColorAt(pointIndex, colors)` — the point's current colour
///     as 0..1 rgba, for colour-driven tints (e.g. `length(rgb)` ramps).
///   * `SCR_TINT_BUF_*` slot constants for kernel `[[buffer(...)]]` attrs.
///
/// The output buffer is `device float4 *tints` (RGBA per point). The
/// ColorPass composes `final = mix(stored.rgb, tint.rgb, tint.a)`, so
/// `tint.a==0` is a pass-through and `tint.a==1` is a full replacement. A
/// **negative** `tint.a` is a discard sentinel: the point is dropped (no colour
/// write), so a pixel covered only by such points resolves fully transparent.
///
/// **Cloud targeting.** ``encode(commandBuffer:cloud:)`` defaults to the
/// first cloud on the rasteriser. Pass `cloud:` explicitly for multi-cloud
/// setups. Encoding is a no-op if no cloud is resolved.
public final class TintPass {
    // MARK: - Buffer slot constants
    //
    // These mirror `SCR_TINT_BUF_*` in the metal preamble. Slots 0–7 are
    // bound by the pass automatically (slot 7 = native colours); slots 8–15
    // are reserved for user uniforms.
    public static let bufferBatches: Int = 0
    public static let bufferXYZLow: Int  = 1
    public static let bufferXYZMed: Int  = 2
    public static let bufferXYZHigh: Int = 3
    public static let bufferLevels: Int  = 4
    public static let bufferTints: Int   = 5
    public static let bufferInfo: Int    = 6
    /// Native per-point colours (packed RGBA8, one `uint` per point), so a tint
    /// kernel can read each point's current colour — e.g. `scr_decodeColorAt`.
    public static let bufferColors: Int  = 7
    public static let bufferUser0: Int   = 8
    public static let bufferUser1: Int   = 9
    public static let bufferUser2: Int   = 10
    public static let bufferUser3: Int   = 11
    public static let bufferUser4: Int   = 12
    public static let bufferUser5: Int   = 13
    public static let bufferUser6: Int   = 14
    public static let bufferUser7: Int   = 15

    /// Bind extra uniforms / buffers each frame. Called inside the compute
    /// encoder after the pass's own bindings, before dispatch. Bind to
    /// ``bufferUser0``…``bufferUser7``.
    public var bindUserBuffers: ((MTLComputeCommandEncoder) -> Void)?

    /// Fired on main after a live-reload rebuilds the pipeline.
    public var onReloaded: (() -> Void)?

    // MARK: - State

    private weak var rasteriser: ComputeRasteriser?
    private let kernelURL: URL
    private let kernelName: String
    private let compiler: MetalFileCompiler
    private var pipeline: MTLComputePipelineState?
    private var infoBuffer: MTLBuffer?
    private var cancellable: AnyCancellable?

    private struct InfoData {
        var pointsPerBatch: UInt32
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }

    // MARK: - Init

    public init(
        rasteriser: ComputeRasteriser,
        kernelURL: URL,
        kernelName: String = "computeTint",
        live: Bool = true
    ) {
        self.rasteriser = rasteriser
        self.kernelURL = kernelURL
        self.kernelName = kernelName
        self.compiler = MetalFileCompiler(watch: live)
        self.infoBuffer = rasteriser.context.device.makeBuffer(
            length: MemoryLayout<InfoData>.stride,
            options: .storageModeShared
        )
        self.infoBuffer?.label = "TintPass.Info"
        rebuildPipeline()
        if live {
            cancellable = compiler.onUpdatePublisher
                .receive(on: DispatchQueue.main)
                .sink { [weak self] in
                    self?.rebuildPipeline()
                    self?.onReloaded?()
                }
        }
    }

    // MARK: - Encode

    /// Per-frame entry point. Auto-targets the first cloud on the rasteriser;
    /// pass `cloud:` for multi-cloud setups. Call before `super.draw(...)`
    /// so the color pass sees this frame's tints.
    public func encode(
        commandBuffer: MTLCommandBuffer,
        cloud: ComputeRasteriserPointCloud? = nil
    ) {
        guard let rasteriser,
              let pipeline,
              let target = cloud ?? rasteriser.pointClouds.first,
              let batchesBuffer = target.batchesBuffer,
              let xyzLow = target.xyzLowBuffer,
              let xyzMed = target.xyzMedBuffer,
              let xyzHigh = target.xyzHighBuffer,
              let levels = target.levelsBuffer,
              let colors = target.colorsBuffer,
              let info = infoBuffer
        else { return }

        if target.tintBuffer == nil {
            target.tintBuffer = target.makeTintBuffer(
                storage: .private,
                label: "TintPass.Tints"
            )
        }
        guard let out = target.tintBuffer else { return }

        if !rasteriser.configuration.applyTint {
            rasteriser.configuration.applyTint = true
        }

        var infoData = InfoData(pointsPerBatch: UInt32(target.capacity.pointsPerBatch))
        memcpy(info.contents(), &infoData, MemoryLayout<InfoData>.size)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "TintPass"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(batchesBuffer, offset: 0, index: Self.bufferBatches)
        encoder.setBuffer(xyzLow,        offset: 0, index: Self.bufferXYZLow)
        encoder.setBuffer(xyzMed,        offset: 0, index: Self.bufferXYZMed)
        encoder.setBuffer(xyzHigh,       offset: 0, index: Self.bufferXYZHigh)
        encoder.setBuffer(levels,        offset: 0, index: Self.bufferLevels)
        encoder.setBuffer(out,           offset: 0, index: Self.bufferTints)
        encoder.setBuffer(info,          offset: 0, index: Self.bufferInfo)
        encoder.setBuffer(colors,        offset: 0, index: Self.bufferColors)
        bindUserBuffers?(encoder)

        let gridSize = target.capacity.maxResidentBatches * target.capacity.pointsPerBatch
        let tew = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: gridSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tew, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Disable the color pass's tint reads. Useful when tearing down the
    /// pass without destroying the cloud (the pass leaves `applyTint = true`
    /// after first encode).
    public func disable() {
        rasteriser?.configuration.applyTint = false
    }

    // MARK: - Pipeline build

    private func rebuildPipeline() {
        guard let rasteriser else { return }
        do {
            let userSource = try compiler.parse(kernelURL)
            let source = Self.preamble + "\n" + userSource
            let library = try rasteriser.context.device.makeLibrary(source: source, options: nil)
            guard let function = library.makeFunction(name: kernelName) else {
                print("[TintPass] kernel '\(kernelName)' not found in \(kernelURL.lastPathComponent)")
                return
            }
            pipeline = try rasteriser.context.device.makeComputePipelineState(function: function)
        } catch {
            print("[TintPass] pipeline build failed: \(error)")
        }
    }

    // MARK: - Preamble
    //
    // Same point-decode + slot-resolve helpers as DisplacementPass. The
    // only difference is the output binding (`device float4 *tints`).
    private static let preamble: String = """
    // Auto-generated tint preamble — SatinComputeRasteriser.

    #include <metal_stdlib>
    using namespace metal;

    #define SCR_STEPS_30BIT 1073741824.0f
    #define SCR_STEPS_20BIT 1048576.0f
    #define SCR_STEPS_10BIT 1024.0f
    #define SCR_MASK_10BIT  1023u

    #define SCR_TINT_BUF_BATCHES   0
    #define SCR_TINT_BUF_XYZ_LOW   1
    #define SCR_TINT_BUF_XYZ_MED   2
    #define SCR_TINT_BUF_XYZ_HIGH  3
    #define SCR_TINT_BUF_LEVELS    4
    #define SCR_TINT_BUF_OUT       5
    #define SCR_TINT_BUF_INFO      6
    #define SCR_TINT_BUF_COLORS    7
    #define SCR_TINT_BUF_USER0     8
    #define SCR_TINT_BUF_USER1     9
    #define SCR_TINT_BUF_USER2     10
    #define SCR_TINT_BUF_USER3     11
    #define SCR_TINT_BUF_USER4     12
    #define SCR_TINT_BUF_USER5     13
    #define SCR_TINT_BUF_USER6     14
    #define SCR_TINT_BUF_USER7     15

    typedef struct {
        int  state;
        float minX, minY, minZ;
        float maxX, maxY, maxZ;
        uint numPoints;
        uint firstPoint;
        uint fileIndex;
        uint scr_padding3, scr_padding4, scr_padding5, scr_padding6, scr_padding7, scr_padding8;
    } RasterBatch;

    typedef struct {
        uint pointsPerBatch;
        uint scr_reserved0;
        uint scr_reserved1;
        uint scr_reserved2;
    } ScrTintInfo;

    inline uint scr_unpack10(uint encoded, uint shift) {
        return (encoded >> shift) & SCR_MASK_10BIT;
    }

    inline float3 scr_decodePointAt(
        uint pointIndex,
        RasterBatch batch,
        device const uint *xyzLow,
        device const uint *xyzMed,
        device const uint *xyzHigh,
        device const uchar *levels)
    {
        const float3 wgMin = float3(batch.minX, batch.minY, batch.minZ);
        const float3 wgSize = max(
            float3(batch.maxX, batch.maxY, batch.maxZ) - wgMin,
            float3(1e-6));
        const int level = int(uint(levels[pointIndex]) & 0x7u);
        if (level == 0) {
            const uint low  = xyzLow[pointIndex];
            const uint med  = xyzMed[pointIndex];
            const uint high = xyzHigh[pointIndex];
            const uint x = (scr_unpack10(low,  0) << 20) | (scr_unpack10(med,  0) << 10) | scr_unpack10(high,  0);
            const uint y = (scr_unpack10(low, 10) << 20) | (scr_unpack10(med, 10) << 10) | scr_unpack10(high, 10);
            const uint z = (scr_unpack10(low, 20) << 20) | (scr_unpack10(med, 20) << 10) | scr_unpack10(high, 20);
            return float3(x, y, z) * (wgSize / SCR_STEPS_30BIT) + wgMin;
        }
        if (level == 1) {
            const uint low = xyzLow[pointIndex];
            const uint med = xyzMed[pointIndex];
            const uint x = (scr_unpack10(low,  0) << 10) | scr_unpack10(med,  0);
            const uint y = (scr_unpack10(low, 10) << 10) | scr_unpack10(med, 10);
            const uint z = (scr_unpack10(low, 20) << 10) | scr_unpack10(med, 20);
            return float3(x, y, z) * (wgSize / SCR_STEPS_20BIT) + wgMin;
        }
        const uint low = xyzLow[pointIndex];
        const uint x = scr_unpack10(low,  0);
        const uint y = scr_unpack10(low, 10);
        const uint z = scr_unpack10(low, 20);
        return float3(x, y, z) * (wgSize / SCR_STEPS_10BIT) + wgMin;
    }

    // Decode a point's native colour (packed RGBA8, little-endian) to 0..1 rgba.
    // `pointIndex` is the same index `scr_resolveTintThread` yields, so this reads
    // the exact colour the ColorPass would otherwise blend into.
    inline float4 scr_decodeColorAt(uint pointIndex, device const uint *colors) {
        const uint c = colors[pointIndex];
        return float4(float( c        & 0xffu),
                      float((c >>  8) & 0xffu),
                      float((c >> 16) & 0xffu),
                      float((c >> 24) & 0xffu)) * (1.0 / 255.0);
    }

    inline bool scr_resolveTintThread(
        uint id,
        constant ScrTintInfo &info,
        device const RasterBatch *batches,
        thread RasterBatch &batch,
        thread uint &pointIndex,
        thread uint &localOffset)
    {
        const uint slot = id / info.pointsPerBatch;
        localOffset = id - slot * info.pointsPerBatch;
        batch = batches[slot];
        if (batch.state == 0 || localOffset >= batch.numPoints) return false;
        pointIndex = batch.firstPoint + localOffset;
        return true;
    }

    #define SCR_TINT_KERNEL_BUFFERS \\
        device const RasterBatch *batches            [[buffer(SCR_TINT_BUF_BATCHES)]], \\
        device const uint        *xyzLow             [[buffer(SCR_TINT_BUF_XYZ_LOW)]], \\
        device const uint        *xyzMed             [[buffer(SCR_TINT_BUF_XYZ_MED)]], \\
        device const uint        *xyzHigh            [[buffer(SCR_TINT_BUF_XYZ_HIGH)]], \\
        device const uchar       *levels             [[buffer(SCR_TINT_BUF_LEVELS)]], \\
        device       float4      *tints              [[buffer(SCR_TINT_BUF_OUT)]], \\
        constant     ScrTintInfo &_scrInfo           [[buffer(SCR_TINT_BUF_INFO)]], \\
        device const uint        *colors             [[buffer(SCR_TINT_BUF_COLORS)]]

    """
}
