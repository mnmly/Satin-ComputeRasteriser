import Metal
import Satin
import simd

public final class ComputeRasteriserPointCloud: Object, @unchecked Sendable {
    public private(set) var packed: PackedPointCloud

    public private(set) var batchesBuffer: MTLBuffer?
    /// Ring-buffered per-batch transforms. Each `updateFiles(...)` advances
    /// to the next slot so multiple updates within a single command-buffer
    /// (stereo: one per eye, or future N-camera setups) write distinct
    /// regions of the buffer. ``filesBufferOffset`` returns the byte offset
    /// of the most recent slot — bind at this offset when encoding.
    public private(set) var filesBuffer: MTLBuffer?
    /// Byte offset of the slot most recently written by ``updateFiles(viewProjection:modelMatrix:frustumTransform:)``.
    /// Bind ``filesBuffer`` at this offset so the GPU reads the matching
    /// per-camera transforms when this encode actually executes.
    public private(set) var filesBufferOffset: Int = 0
    public private(set) var xyzLowBuffer: MTLBuffer?
    public private(set) var xyzMedBuffer: MTLBuffer?
    public private(set) var xyzHighBuffer: MTLBuffer?
    public private(set) var colorsBuffer: MTLBuffer?
    public private(set) var levelsBuffer: MTLBuffer?
    public private(set) var visibleBatchesBuffer: MTLBuffer?
    public private(set) var cullCounterBuffer: MTLBuffer?
    public private(set) var cullIndirectArgsBuffer: MTLBuffer?

    /// Optional per-point displacement (one `float3` per pack-order point,
    /// stride 16 bytes). Bound at `Custom8` on the depth + color passes when
    /// `ComputeRasteriserConfiguration.applyDisplacement == true`.
    public var displacementBuffer: MTLBuffer?

    /// Allocate a `pointCount * stride(float3)` buffer suitable for use as
    /// `displacementBuffer`. Caller owns the buffer; fill it from a compute
    /// kernel each frame (or once, for static displacement).
    public func makeDisplacementBuffer(
        storage: MTLStorageMode = .private,
        label: String? = nil
    ) -> MTLBuffer? {
        guard pointCount > 0 else { return nil }
        let length = pointCount * MemoryLayout<SIMD3<Float>>.stride
        let buffer = context.device.makeBuffer(length: length, options: storage == .private ? .storageModePrivate : .storageModeShared)
        buffer?.label = label ?? "\(self.label).Displacement"
        return buffer
    }

    public var batchCount: Int { packed.batchCount }
    public var pointCount: Int { packed.pointCount }

    /// Number of in-flight `filesBuffer` slots. Sized for stereo (2 cameras
    /// per frame) × Satin's triple-buffering (3 frames in flight) = 6.
    /// Filesbuffer per-slot stride is small (≈256 B per RasterFile × file
    /// count, typically a handful), so the extra memory is trivial.
    public static let filesBufferSlotCount: Int = Satin.maxBuffersInFlight * 2

    private var filesSlotIndex: Int = -1
    private var filesSlotStride: Int = 0

    /// `BindableBuffer` view onto the most recently written ``filesBuffer``
    /// slot. Pass this (not `filesBuffer` directly) to compute processors so
    /// Satin's bind path reads from the correct ring offset. Sequential
    /// `updateFiles` calls within one command buffer advance the slot, so
    /// each encode's bound view points at its own per-camera transforms.
    public var filesBindable: (any BindableBuffer)? {
        guard let filesBuffer else { return nil }
        return FilesSlotView(buffer: filesBuffer, offset: filesBufferOffset)
    }

    private struct FilesSlotView: BindableBuffer {
        let buffer: MTLBuffer!
        let offset: Int
    }

    public init(context: Context, packed: PackedPointCloud, label: String = "ComputeRasteriserPointCloud") {
        self.packed = packed
        super.init(context: context, label: label)
        rebuildBuffers()
    }

    public required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    public func replacePackedPointCloud(_ packed: PackedPointCloud) {
        self.packed = packed
        rebuildBuffers()
    }

    public func updateFiles(
        viewProjection: simd_float4x4,
        modelMatrix: simd_float4x4,
        frustumTransform: simd_float4x4? = nil
    ) {
        guard !packed.files.isEmpty, let filesBuffer else { return }
        var files = packed.files
        for index in files.indices {
            let world = modelMatrix * files[index].world
            files[index].transform = viewProjection * world
            files[index].transformFrustum = (frustumTransform ?? viewProjection) * world
            files[index].world = world
        }

        filesSlotIndex = (filesSlotIndex + 1) % Self.filesBufferSlotCount
        filesBufferOffset = filesSlotIndex * filesSlotStride

        let byteCount = min(filesSlotStride, files.count * MemoryLayout<RasterFile>.stride)
        files.withUnsafeBytes { bytes in
            filesBuffer.contents()
                .advanced(by: filesBufferOffset)
                .copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
    }

    private func rebuildBuffers() {
        batchesBuffer = makeBuffer(packed.batches, label: "\(label).Batches")
        filesBuffer = makeFilesRingBuffer(files: packed.files, label: "\(label).Files")
        filesSlotIndex = -1
        filesBufferOffset = 0
        xyzLowBuffer = makeBuffer(packed.xyzLow, label: "\(label).XYZLow")
        xyzMedBuffer = makeBuffer(packed.xyzMed, label: "\(label).XYZMed")
        xyzHighBuffer = makeBuffer(packed.xyzHigh, label: "\(label).XYZHigh")
        colorsBuffer = makeBuffer(packed.colors, label: "\(label).Colors")
        levelsBuffer = makeBuffer(packed.levels, label: "\(label).Levels")
        rebuildCullBuffers()
    }

    private func rebuildCullBuffers() {
        guard packed.batchCount > 0 else {
            visibleBatchesBuffer = nil
            cullCounterBuffer = nil
            cullIndirectArgsBuffer = nil
            return
        }
        visibleBatchesBuffer = context.device.makeBuffer(
            length: packed.batchCount * MemoryLayout<VisibleBatch>.stride,
            options: .storageModePrivate
        )
        visibleBatchesBuffer?.label = "\(label).VisibleBatches"
        cullCounterBuffer = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModePrivate
        )
        cullCounterBuffer?.label = "\(label).CullCounter"
        cullIndirectArgsBuffer = context.device.makeBuffer(
            length: MemoryLayout<CRDispatchArgs>.stride,
            options: .storageModePrivate
        )
        cullIndirectArgsBuffer?.label = "\(label).CullIndirectArgs"
    }

    private func makeBuffer<T>(_ values: [T], label: String) -> MTLBuffer? {
        guard !values.isEmpty else { return nil }
        let byteCount = values.count * MemoryLayout<T>.stride
        let buffer = values.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: byteCount, options: .storageModeShared)
        }
        buffer?.label = label
        return buffer
    }

    /// `filesBuffer` is sized for N in-flight slots so sequential
    /// `updateFiles` calls within one command buffer don't clobber each
    /// other. Slot 0 is seeded with `files` so the first encode that runs
    /// before `updateFiles` has been called sees the identity-world state.
    private func makeFilesRingBuffer(files: [RasterFile], label: String) -> MTLBuffer? {
        guard !files.isEmpty else { return nil }
        let perSlot = files.count * MemoryLayout<RasterFile>.stride
        filesSlotStride = perSlot
        let totalLength = perSlot * Self.filesBufferSlotCount
        guard let buffer = context.device.makeBuffer(length: totalLength, options: .storageModeShared) else { return nil }
        buffer.label = label
        files.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: perSlot)
        }
        return buffer
    }

    private func updateBuffer<T>(_ buffer: MTLBuffer?, with values: [T]) {
        guard let buffer, !values.isEmpty else { return }
        let byteCount = min(buffer.length, values.count * MemoryLayout<T>.stride)
        values.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
    }
}
