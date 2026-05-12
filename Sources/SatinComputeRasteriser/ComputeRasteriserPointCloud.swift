import Metal
import Satin
import simd

public final class ComputeRasteriserPointCloud: Object, @unchecked Sendable {
    public private(set) var packed: PackedPointCloud

    public private(set) var batchesBuffer: MTLBuffer?
    public private(set) var filesBuffer: MTLBuffer?
    public private(set) var xyzLowBuffer: MTLBuffer?
    public private(set) var xyzMedBuffer: MTLBuffer?
    public private(set) var xyzHighBuffer: MTLBuffer?
    public private(set) var colorsBuffer: MTLBuffer?

    public var batchCount: Int { packed.batchCount }
    public var pointCount: Int { packed.pointCount }

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
        guard !packed.files.isEmpty else { return }
        var files = packed.files
        for index in files.indices {
            let world = modelMatrix * files[index].world
            files[index].transform = viewProjection * world
            files[index].transformFrustum = (frustumTransform ?? viewProjection) * world
            files[index].world = world
        }
        updateBuffer(filesBuffer, with: files)
    }

    private func rebuildBuffers() {
        batchesBuffer = makeBuffer(packed.batches, label: "\(label).Batches")
        filesBuffer = makeBuffer(packed.files, label: "\(label).Files")
        xyzLowBuffer = makeBuffer(packed.xyzLow, label: "\(label).XYZLow")
        xyzMedBuffer = makeBuffer(packed.xyzMed, label: "\(label).XYZMed")
        xyzHighBuffer = makeBuffer(packed.xyzHigh, label: "\(label).XYZHigh")
        colorsBuffer = makeBuffer(packed.colors, label: "\(label).Colors")
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

    private func updateBuffer<T>(_ buffer: MTLBuffer?, with values: [T]) {
        guard let buffer, !values.isEmpty else { return }
        let byteCount = min(buffer.length, values.count * MemoryLayout<T>.stride)
        values.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
    }
}
