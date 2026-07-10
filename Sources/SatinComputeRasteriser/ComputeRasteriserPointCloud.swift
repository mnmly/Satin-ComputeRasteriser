import Foundation
import Metal
import Satin
import simd

/// Sizing for a ``ComputeRasteriserPointCloud``'s slot pool.
///
/// The renderer pre-allocates GPU buffers for `maxResidentBatches` slots,
/// each `pointsPerBatch` points wide. Streaming sources (e.g.
/// SwiftPDAL's `CopcStreamingPointCloudSource`) page chunks in and out
/// of those slots without touching the underlying buffers — uploads are
/// `MTLBlitCommandEncoder` copies into a slot's byte range.
///
/// Total VRAM cost for positions + colors + levels:
///
/// ```
/// maxResidentBatches × pointsPerBatch × 17 bytes
/// ```
///
/// At the defaults below (`8192 × 10240`) that's ~1.4 GB. Tune
/// `maxResidentBatches` to your VRAM budget and the typical working set
/// of your camera path.
public struct ComputeRasteriserCapacity: Hashable, Sendable {
    /// Number of fixed-size slots in the pool. Each slot holds at most
    /// `pointsPerBatch` points and one ``RasterBatch`` of metadata.
    public let maxResidentBatches: Int
    /// Per-slot point capacity. Must match the streaming source's
    /// `pointsPerBatch` (SwiftPDAL's default is `128 * 80 = 10240`).
    public let pointsPerBatch: Int

    public init(maxResidentBatches: Int, pointsPerBatch: Int = computeRasteriserThreadsPerGroup * 80) {
        precondition(maxResidentBatches > 0, "maxResidentBatches must be positive")
        precondition(pointsPerBatch > 0, "pointsPerBatch must be positive")
        self.maxResidentBatches = maxResidentBatches
        self.pointsPerBatch = pointsPerBatch
    }

    /// Total point capacity of the pool (`maxResidentBatches × pointsPerBatch`).
    public var maxResidentPoints: Int { maxResidentBatches * pointsPerBatch }
}

/// A point cloud rendered by ``ComputeRasteriser``.
///
/// Owns a fixed-size slot pool of GPU buffers (positions split into 3×10-bit
/// `UInt32`, packed RGBA `UInt32`, per-point `UInt8` LOD level, and one
/// ``RasterBatch`` of metadata per slot). Two ways to populate it:
///
/// * **Wholesale**, via ``init(context:packed:label:)`` /
///   ``replacePackedPointCloud(_:)`` — sizes the pool to fit and uploads
///   in one shot. Backwards-compatible with pre-streaming code.
/// * **Incremental**, via ``init(context:capacity:files:originShift:label:)``
///   followed by ``addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
///   and ``removeBatches(slots:commit:)`` — designed for an external streaming
///   source (e.g. SwiftPDAL) that pages chunks in and out as the camera moves.
///
/// In both modes the cull kernel iterates every slot and short-circuits
/// non-resident ones via ``RasterBatch/state``, so empty slots cost one
/// branch per frame.
public final class ComputeRasteriserPointCloud: Object, @unchecked Sendable {
    /// Slot-pool sizing the cloud was created with. May change if a
    /// ``replacePackedPointCloud(_:)`` call needs to grow the pool.
    public private(set) var capacity: ComputeRasteriserCapacity

    /// Per-file transforms (model/view/projection bake-out). The first slot
    /// is what the renderer projects through; non-streaming users see one
    /// identity-world `RasterFile`. Streaming clouds get an `originShift`
    /// translation baked into `files[0].world` at init.
    public private(set) var files: [RasterFile]

    public private(set) var batchesBuffer: MTLBuffer?
    /// Ring-buffered per-batch transforms. Each `updateFiles(...)` advances
    /// to the next slot so multiple updates within a single command-buffer
    /// (stereo: one per eye, or future N-camera setups) write distinct
    /// regions of the buffer. ``filesBufferOffset`` returns the byte offset
    /// of the most recent slot — bind at this offset when encoding.
    public private(set) var filesBuffer: MTLBuffer?
    /// Byte offset of the slot most recently written by
    /// ``updateFiles(viewProjection:modelMatrix:frustumTransform:)``.
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

    /// Optional per-point displacement (one `float3` per slot-pool point,
    /// stride 16 bytes). Bound at `Custom8` on the depth + color passes when
    /// `ComputeRasteriserConfiguration.applyDisplacement == true`.
    public var displacementBuffer: MTLBuffer?

    /// Previous frame's `displacementBuffer`, held for motion blur so the color
    /// pass can compute per-point displacement velocity. Keyed by the rendering
    /// camera's `ObjectIdentifier` so STEREO keeps one per eye (each eye renders
    /// in its own drained command buffer and snapshots its own copy). The
    /// rasteriser blits `displacementBuffer → prevDisplacementBuffers[cam]` at the
    /// end of each frame (only while `configuration.motionBlur > 0`). Bound at raw
    /// buffer index 12 on the color pass (an otherwise-unused compute slot).
    public var prevDisplacementBuffers: [ObjectIdentifier: MTLBuffer] = [:]

    /// Optional per-point color tint (one `float4` per slot-pool point,
    /// stride 16 bytes). Bound at `Custom10` on the color pass when
    /// `ComputeRasteriserConfiguration.applyTint == true`. The color pass
    /// composes each point's stored RGB with `tint.rgb` weighted by
    /// `tint.a`: `final = mix(original, tint.rgb, tint.a)`. So `tint.a==0`
    /// is a pass-through, `tint.a==1` is a full replacement, and partial
    /// alphas modulate.
    public var tintBuffer: MTLBuffer?

    /// Allocate a `capacity.maxResidentPoints * stride(float3)` buffer
    /// suitable for use as ``displacementBuffer``. Sized by pool capacity
    /// (not current resident count) so the buffer is allocated once and
    /// addressed by `firstPoint + localOffset` regardless of which slots
    /// are resident.
    public func makeDisplacementBuffer(
        storage: MTLStorageMode = .private,
        label: String? = nil
    ) -> MTLBuffer? {
        let length = capacity.maxResidentPoints * MemoryLayout<SIMD3<Float>>.stride
        let buffer = context.device.makeBuffer(length: length, options: storage == .private ? .storageModePrivate : .storageModeShared)
        buffer?.label = label ?? "\(self.label).Displacement"
        return buffer
    }

    /// Allocate a `capacity.maxResidentPoints * stride(float4)` buffer
    /// suitable for use as ``tintBuffer``. Same sizing semantics as
    /// ``makeDisplacementBuffer(storage:label:)``.
    public func makeTintBuffer(
        storage: MTLStorageMode = .private,
        label: String? = nil
    ) -> MTLBuffer? {
        let length = capacity.maxResidentPoints * MemoryLayout<SIMD4<Float>>.stride
        let buffer = context.device.makeBuffer(length: length, options: storage == .private ? .storageModePrivate : .storageModeShared)
        buffer?.label = label ?? "\(self.label).Tint"
        return buffer
    }

    /// Number of slots the cull kernel dispatches over (= ``capacity``'s
    /// `maxResidentBatches`). Non-resident slots are skipped before the
    /// frustum test via ``RasterBatch/state`` `== 0`.
    public var batchCount: Int { capacity.maxResidentBatches }

    /// Number of slots currently holding a resident batch.
    public private(set) var residentBatchCount: Int = 0

    /// Sum of `numPoints` across all resident batches.
    public private(set) var pointCount: Int = 0

    /// Number of in-flight ``filesBuffer`` slots. Sized for stereo (2 cameras
    /// per frame) × Satin's triple-buffering (3 frames in flight) = 6.
    /// Per-slot stride is small (≈256 B per `RasterFile` × file count, typically
    /// a handful), so the extra memory is trivial.
    public static let filesBufferSlotCount: Int = Satin.maxBuffersInFlight * 2

    private var filesSlotIndex: Int = -1
    private var filesSlotStride: Int = 0

    /// CPU-side mirror of every slot's ``RasterBatch``. Mutated by
    /// ``addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
    /// and ``removeBatches(slots:commit:)``, then memcpy'd into ``batchesBuffer``
    /// in one shot. Empty slots have `state == 0` and do not contribute to
    /// ``pointCount`` / ``residentBatchCount``.
    private var batchMirror: [RasterBatch]
    /// Inclusive slot range mutated since the last ``flushBatchMirror()``;
    /// empty (`lo > hi`) means the GPU buffer is up to date.
    private var dirtySlotLo: Int = .max
    private var dirtySlotHi: Int = -1
    /// LIFO of free slot indices. Newest-freed wins so cache lines stay warm.
    private var freeSlots: [Int]

    // Processors read ``filesBuffer`` + ``filesBufferOffset`` directly and
    // bind the offset via their own `applyAdditionalBindings`. Sequential
    // `updateFiles` calls within one command buffer advance the slot, so each
    // encode's binding points at its own per-camera transforms.

    /// Streaming-mode init. Pre-allocates an empty slot pool; the caller
    /// drives residency via
    /// ``addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``
    /// and ``removeBatches(slots:commit:)``.
    ///
    /// - Parameters:
    ///   - context: Satin context (provides the Metal device).
    ///   - capacity: pool size; see ``ComputeRasteriserCapacity``.
    ///   - files: per-file transform table. Streaming sources typically pass
    ///     `[RasterFile(world: translation(originShift))]` so chunk positions
    ///     (which the source pre-shifted to the file's bounds center for
    ///     `Float` precision) land at their original world coordinates.
    ///   - originShift: convenience translation baked into `files[0].world`
    ///     when `files` is left at its default. Ignored if `files` is non-default.
    ///     **Precision warning:** large values (≥ ~10⁵) cause FP32 precision
    ///     loss in the `viewMatrix * world` product because both the camera
    ///     and the file translation end up in absolute world space. For
    ///     georeferenced clouds (LAS/LAZ in projected CRS), prefer leaving
    ///     this `.zero` and rendering in pre-shifted source coordinates;
    ///     the streaming source already pre-shifts chunk positions to be
    ///     near origin.
    ///   - label: human-readable label, used as the prefix on Metal buffer labels.
    public init(
        context: Context,
        capacity: ComputeRasteriserCapacity,
        files: [RasterFile]? = nil,
        originShift: SIMD3<Float> = .zero,
        label: String = "ComputeRasteriserPointCloud"
    ) {
        self.capacity = capacity
        self.files = files ?? [RasterFile(world: matrix_translation(originShift))]
        self.batchMirror = Array(
            repeating: RasterBatch(min: .zero, max: .zero, numPoints: 0, firstPoint: 0, fileIndex: 0, state: 0),
            count: capacity.maxResidentBatches
        )
        self.freeSlots = (0 ..< capacity.maxResidentBatches).reversed()
        super.init(context: context, label: label)
        allocateBuffers()
    }

    /// Convenience init: capacity is sized to fit `packed` exactly and the
    /// cloud is immediately populated. Drop-in for non-streaming code.
    public convenience init(
        context: Context,
        packed: PackedPointCloud,
        label: String = "ComputeRasteriserPointCloud"
    ) {
        let pointsPerBatch = max(1, Int(packed.batches.map(\.numPoints).max() ?? UInt32(computeRasteriserThreadsPerGroup * 80)))
        let cap = ComputeRasteriserCapacity(
            maxResidentBatches: max(1, packed.batchCount),
            pointsPerBatch: pointsPerBatch
        )
        self.init(context: context, capacity: cap, files: packed.files, label: label)
        if packed.batchCount > 0 {
            uploadPacked(packed)
        }
    }

    public required init(from decoder: any Decoder) throws {
        fatalError("init(from:) has not been implemented")
    }

    /// Wholesale replace the cloud's contents. Reallocates the slot pool if
    /// `packed` doesn't fit the current ``capacity`` (preserving the existing
    /// `pointsPerBatch`). Equivalent to `clearAllBatches()` + `addBatches(...)`
    /// for the common case where the new cloud fits.
    public func replacePackedPointCloud(_ packed: PackedPointCloud) {
        let needed = max(1, packed.batchCount)
        let pointsPerBatch = max(1, Int(packed.batches.map(\.numPoints).max() ?? UInt32(capacity.pointsPerBatch)))
        if needed > capacity.maxResidentBatches || pointsPerBatch > capacity.pointsPerBatch {
            // Reallocate.
            let newCapacity = ComputeRasteriserCapacity(
                maxResidentBatches: needed,
                pointsPerBatch: max(pointsPerBatch, capacity.pointsPerBatch)
            )
            reallocate(to: newCapacity, files: packed.files)
        } else {
            files = packed.files
            rebuildFilesBuffer()
            clearAllBatches()
        }
        if packed.batchCount > 0 {
            uploadPacked(packed)
        }
    }

    // MARK: - GPU wholesale pack support

    /// Prepare the slot pool for a wholesale GPU pack of `pointCount` points.
    /// Grows the pool if needed (preserving `pointsPerBatch`), then marks slots
    /// `0..<numBatches` resident with `numPoints`/`firstPoint`/`state` set on
    /// the CPU mirror; per-batch `min`/`max` are left zero here and written by
    /// the GPU directly into ``batchesBuffer``. Returns the number of batches.
    ///
    /// The renderer reads ``batchesBuffer`` (which the GPU fills), so rendering
    /// is correct as soon as the pack command buffer completes — even before
    /// ``adoptGPUBatchBounds(numBatches:)`` re-syncs the CPU mirror.
    @discardableResult
    internal func prepareForGPUPack(pointCount: Int) -> Int {
        let ppb = capacity.pointsPerBatch
        let numBatches = max(1, (pointCount + ppb - 1) / ppb)
        if numBatches > capacity.maxResidentBatches {
            reallocate(
                to: ComputeRasteriserCapacity(maxResidentBatches: numBatches, pointsPerBatch: ppb),
                files: files
            )
        }
        for slot in 0 ..< capacity.maxResidentBatches {
            if slot < numBatches {
                let first = slot * ppb
                let num = min(ppb, pointCount - first)
                batchMirror[slot] = RasterBatch(
                    min: .zero, max: .zero,
                    numPoints: UInt32(num), firstPoint: UInt32(first),
                    fileIndex: 0, state: 1
                )
            } else {
                batchMirror[slot].state = 0
                batchMirror[slot].numPoints = 0
            }
        }
        freeSlots = (numBatches ..< capacity.maxResidentBatches).reversed()
        residentBatchCount = numBatches
        self.pointCount = pointCount
        markAllSlotsDirty()
        flushBatchMirror()
        return numBatches
    }

    /// Copy the GPU-written per-batch `RasterBatch` structs (including the AABBs
    /// the GPU computed) back into the CPU mirror, so later mirror flushes don't
    /// clobber them. Call from a command-buffer completion handler after a GPU
    /// pack, or after `waitUntilCompleted`. ``batchesBuffer`` must be shared.
    internal func adoptGPUBatchBounds(numBatches: Int) {
        guard let batchesBuffer, numBatches > 0 else { return }
        let n = min(numBatches, capacity.maxResidentBatches)
        let ptr = batchesBuffer.contents().bindMemory(to: RasterBatch.self, capacity: n)
        for slot in 0 ..< n { batchMirror[slot] = ptr[slot] }
        markSlotsDirty(0 ..< n)
    }

    /// Reserve a **contiguous** run of `n` free slots for a GPU chunk pack,
    /// returning the first slot index (or `nil` if no run of `n` free slots
    /// exists). The slots are removed from the free list and marked `state = 1`
    /// provisionally; the GPU pack overwrites them with real per-batch data and
    /// ``finalizeGPUChunk(firstSlot:numBatches:count:)`` syncs the mirror.
    ///
    /// A chunk's points are written as one contiguous run (`firstSlot*ppb ...`),
    /// so its batches must occupy adjacent slots — unlike ``addBatches`` which
    /// places each batch independently.
    internal func reserveContiguousSlots(_ n: Int) -> Int? {
        guard n > 0, n <= capacity.maxResidentBatches else { return nil }
        var s = 0
        while s + n <= capacity.maxResidentBatches {
            var k = 0
            while k < n, batchMirror[s + k].state == 0 { k += 1 }
            if k == n {
                let reserved = Set(s ..< (s + n))
                freeSlots.removeAll { reserved.contains($0) }
                for slot in s ..< (s + n) { batchMirror[slot].state = 1 }
                markSlotsDirty(s ..< (s + n))
                return s
            }
            s += k + 1 // slot s+k is occupied → next possible run starts past it
        }
        return nil
    }

    /// After a GPU chunk pack into `[firstSlot, firstSlot+numBatches)`, pull the
    /// GPU-written `RasterBatch` structs into the CPU mirror and account the
    /// points. Caller flushes (e.g. ``commitBatchUpdates()``) once per frame.
    internal func finalizeGPUChunk(firstSlot: Int, numBatches: Int, count: Int) {
        if let batchesBuffer {
            let ptr = batchesBuffer.contents().bindMemory(
                to: RasterBatch.self, capacity: capacity.maxResidentBatches
            )
            for slot in firstSlot ..< min(firstSlot + numBatches, capacity.maxResidentBatches) {
                batchMirror[slot] = ptr[slot]
            }
            markSlotsDirty(firstSlot ..< min(firstSlot + numBatches, capacity.maxResidentBatches))
        }
        residentBatchCount += numBatches
        pointCount += count
    }

    /// Mark every slot empty without freeing GPU buffers. Cheap; subsequent
    /// `addBatches` reuses slot 0 first.
    public func clearAllBatches() {
        for slot in 0 ..< capacity.maxResidentBatches {
            batchMirror[slot].state = 0
            batchMirror[slot].numPoints = 0
        }
        freeSlots = (0 ..< capacity.maxResidentBatches).reversed()
        residentBatchCount = 0
        pointCount = 0
        markAllSlotsDirty()
        flushBatchMirror()
    }

    /// Upload one or more batches into free slots.
    ///
    /// Each input `batches[i]` describes a contiguous run of points in the
    /// position/color/level `Data` blobs starting at `batches[i].firstPoint`.
    /// The renderer rebases that run into a fresh slot: the returned
    /// `slotIndices[i]` × `pointsPerBatch` becomes the slot's `firstPoint`
    /// in the GPU buffers.
    ///
    /// - Parameters:
    ///   - positionsXYZLow: contiguous `UInt32`-per-point low-bits axis pack;
    ///     must cover at least `batches.last.firstPoint + numPoints` entries.
    ///   - positionsXYZMed: middle-bits axis pack, same shape as `positionsXYZLow`.
    ///   - positionsXYZHigh: high-bits axis pack, same shape as `positionsXYZLow`.
    ///   - colors: contiguous packed RGBA `UInt32`-per-point.
    ///   - levels: contiguous `UInt8`-per-point LOD level.
    ///   - batches: metadata, one per chunk to add. Caller's `firstPoint`/`state`
    ///     are rewritten by the renderer; AABB / `numPoints` / `fileIndex`
    ///     are preserved.
    /// - Returns: slot index (in `[0, capacity.maxResidentBatches)`) chosen
    ///   for each input batch. Map this back to your chunk identity so a later
    ///   ``removeBatches(slots:commit:)`` call can free the right slots.
    /// - Precondition: `freeSlotCount >= batches.count`. Call
    ///   ``removeBatches(slots:commit:)`` first if you'd otherwise overflow.
    /// - Parameter commit: if `true` (default), the GPU-visible
    ///   ``batchesBuffer`` is re-uploaded before this call returns. Pass
    ///   `false` when making many sequential calls (e.g. draining a
    ///   streaming source's per-frame delta) and call
    ///   ``commitBatchUpdates()`` once at the end — the per-call upload
    ///   covers the contiguous slot range dirtied since the last flush,
    ///   so coalescing still wins when adds land in scattered slots.
    @discardableResult
    public func addBatches(
        positionsXYZLow: Data,
        positionsXYZMed: Data,
        positionsXYZHigh: Data,
        colors: Data,
        levels: Data,
        batches: [RasterBatch],
        commit: Bool = true
    ) -> [Int] {
        precondition(batches.count <= freeSlots.count, "addBatches: not enough free slots (have \(freeSlots.count), need \(batches.count))")
        guard !batches.isEmpty else { return [] }

        var assigned: [Int] = []
        assigned.reserveCapacity(batches.count)

        for batch in batches {
            let slot = freeSlots.removeLast()
            assigned.append(slot)

            let dstFirstPoint = slot * capacity.pointsPerBatch
            let srcFirstPoint = Int(batch.firstPoint)
            let count = Int(batch.numPoints)
            precondition(count <= capacity.pointsPerBatch, "batch numPoints (\(count)) exceeds slot capacity (\(capacity.pointsPerBatch))")

            copySlice(positionsXYZLow,  srcOffsetPoints: srcFirstPoint, dstOffsetPoints: dstFirstPoint, count: count, stride: 4, into: xyzLowBuffer)
            copySlice(positionsXYZMed,  srcOffsetPoints: srcFirstPoint, dstOffsetPoints: dstFirstPoint, count: count, stride: 4, into: xyzMedBuffer)
            copySlice(positionsXYZHigh, srcOffsetPoints: srcFirstPoint, dstOffsetPoints: dstFirstPoint, count: count, stride: 4, into: xyzHighBuffer)
            copySlice(colors,           srcOffsetPoints: srcFirstPoint, dstOffsetPoints: dstFirstPoint, count: count, stride: 4, into: colorsBuffer)
            copySlice(levels,           srcOffsetPoints: srcFirstPoint, dstOffsetPoints: dstFirstPoint, count: count, stride: 1, into: levelsBuffer)

            var slotBatch = batch
            slotBatch.firstPoint = UInt32(dstFirstPoint)
            slotBatch.state = 1
            batchMirror[slot] = slotBatch
            markSlotDirty(slot)

            residentBatchCount += 1
            pointCount += count
        }

        if commit { flushBatchMirror() }
        return assigned
    }

    /// Free the given slots. Their ``RasterBatch/state`` flips to `0` so the
    /// cull kernel skips them next frame; the slots return to the free list
    /// and may be reused by a subsequent `addBatches`.
    /// - Parameters:
    ///   - slots: Indices previously returned by `addBatches`.
    ///   - commit: see ``addBatches(positionsXYZLow:positionsXYZMed:positionsXYZHigh:colors:levels:batches:commit:)``.
    public func removeBatches(slots: [Int], commit: Bool = true) {
        guard !slots.isEmpty else { return }
        for slot in slots {
            precondition(slot >= 0 && slot < capacity.maxResidentBatches, "removeBatches: slot \(slot) out of range")
            if batchMirror[slot].state == 0 { continue }
            pointCount -= Int(batchMirror[slot].numPoints)
            residentBatchCount -= 1
            batchMirror[slot].state = 0
            batchMirror[slot].numPoints = 0
            markSlotDirty(slot)
            freeSlots.append(slot)
        }
        if commit { flushBatchMirror() }
    }

    /// Re-upload the CPU-side batch mirror to the GPU-visible
    /// ``batchesBuffer``. Use after a series of `commit: false`
    /// add/remove calls to publish all changes in one shot.
    ///
    /// Only the contiguous slot range dirtied since the last flush is
    /// copied; a no-op when nothing changed.
    public func commitBatchUpdates() {
        flushBatchMirror()
    }

    /// Per-frame cycle of the ``filesBuffer`` ring + memcpy of the latest
    /// transforms. Called by the rasteriser once per camera per encode.
    public func updateFiles(
        viewProjection: simd_float4x4,
        modelMatrix: simd_float4x4,
        frustumTransform: simd_float4x4? = nil,
        prevViewProjection: simd_float4x4? = nil
    ) {
        guard !files.isEmpty, let filesBuffer else { return }
        var snapshot = files
        for index in snapshot.indices {
            let world = modelMatrix * files[index].world
            snapshot[index].transform = viewProjection * world
            snapshot[index].transformFrustum = (frustumTransform ?? viewProjection) * world
            snapshot[index].world = world
            // Previous-frame MVP for motion-blur velocity. Uses the current world
            // (objects are effectively static frame-to-frame; camera motion, the
            // dominant term, comes from prevViewProjection).
            snapshot[index].prevTransform = (prevViewProjection ?? viewProjection) * world
        }

        filesSlotIndex = (filesSlotIndex + 1) % Self.filesBufferSlotCount
        filesBufferOffset = filesSlotIndex * filesSlotStride

        let byteCount = min(filesSlotStride, snapshot.count * MemoryLayout<RasterFile>.stride)
        snapshot.withUnsafeBytes { bytes in
            filesBuffer.contents()
                .advanced(by: filesBufferOffset)
                .copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
    }

    // MARK: - Pool plumbing

    private func reallocate(to newCapacity: ComputeRasteriserCapacity, files: [RasterFile]) {
        capacity = newCapacity
        self.files = files
        self.batchMirror = Array(
            repeating: RasterBatch(min: .zero, max: .zero, numPoints: 0, firstPoint: 0, fileIndex: 0, state: 0),
            count: newCapacity.maxResidentBatches
        )
        self.freeSlots = (0 ..< newCapacity.maxResidentBatches).reversed()
        self.residentBatchCount = 0
        self.pointCount = 0
        allocateBuffers()
    }

    private func allocateBuffers() {
        let pointsCap = capacity.maxResidentPoints
        xyzLowBuffer  = makeEmptyBuffer(length: pointsCap * 4, label: "\(label).XYZLow")
        xyzMedBuffer  = makeEmptyBuffer(length: pointsCap * 4, label: "\(label).XYZMed")
        xyzHighBuffer = makeEmptyBuffer(length: pointsCap * 4, label: "\(label).XYZHigh")
        colorsBuffer  = makeEmptyBuffer(length: pointsCap * 4, label: "\(label).Colors")
        levelsBuffer  = makeEmptyBuffer(length: pointsCap,     label: "\(label).Levels")

        let batchesByteCount = capacity.maxResidentBatches * MemoryLayout<RasterBatch>.stride
        batchesBuffer = makeEmptyBuffer(length: batchesByteCount, label: "\(label).Batches")
        markAllSlotsDirty()
        flushBatchMirror()

        rebuildFilesBuffer()
        rebuildCullBuffers()
    }

    private func rebuildFilesBuffer() {
        guard !files.isEmpty else {
            filesBuffer = nil
            filesBufferOffset = 0
            filesSlotIndex = -1
            filesSlotStride = 0
            return
        }
        let perSlot = files.count * MemoryLayout<RasterFile>.stride
        filesSlotStride = perSlot
        let totalLength = perSlot * Self.filesBufferSlotCount
        guard let buffer = context.device.makeBuffer(length: totalLength, options: .storageModeShared) else {
            filesBuffer = nil
            return
        }
        buffer.label = "\(label).Files"
        files.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: perSlot)
        }
        filesBuffer = buffer
        filesSlotIndex = -1
        filesBufferOffset = 0
    }

    private func rebuildCullBuffers() {
        visibleBatchesBuffer = context.device.makeBuffer(
            length: capacity.maxResidentBatches * MemoryLayout<VisibleBatch>.stride,
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

    private func uploadPacked(_ packed: PackedPointCloud) {
        let positionLow = Data(bytes: packed.xyzLow, count: packed.xyzLow.count * 4)
        let positionMed = Data(bytes: packed.xyzMed, count: packed.xyzMed.count * 4)
        let positionHigh = Data(bytes: packed.xyzHigh, count: packed.xyzHigh.count * 4)
        let colors = Data(bytes: packed.colors, count: packed.colors.count * 4)
        let levels = Data(packed.levels)
        addBatches(
            positionsXYZLow: positionLow,
            positionsXYZMed: positionMed,
            positionsXYZHigh: positionHigh,
            colors: colors,
            levels: levels,
            batches: packed.batches
        )
    }

    private func makeEmptyBuffer(length: Int, label: String) -> MTLBuffer? {
        guard length > 0 else { return nil }
        let buffer = context.device.makeBuffer(length: length, options: .storageModeShared)
        buffer?.label = label
        return buffer
    }

    private func markSlotDirty(_ slot: Int) {
        if slot < dirtySlotLo { dirtySlotLo = slot }
        if slot > dirtySlotHi { dirtySlotHi = slot }
    }

    private func markSlotsDirty(_ range: Range<Int>) {
        guard !range.isEmpty else { return }
        markSlotDirty(range.lowerBound)
        markSlotDirty(range.upperBound - 1)
    }

    private func markAllSlotsDirty() {
        markSlotsDirty(0 ..< batchMirror.count)
    }

    private func flushBatchMirror() {
        guard let batchesBuffer, dirtySlotLo <= dirtySlotHi else { return }
        let stride = MemoryLayout<RasterBatch>.stride
        let lo = max(0, dirtySlotLo)
        let hi = min(dirtySlotHi, batchMirror.count - 1, batchesBuffer.length / stride - 1)
        dirtySlotLo = .max
        dirtySlotHi = -1
        guard lo <= hi else { return }
        let byteOffset = lo * stride
        let byteCount = (hi - lo + 1) * stride
        batchMirror.withUnsafeBytes { bytes in
            batchesBuffer.contents()
                .advanced(by: byteOffset)
                .copyMemory(from: bytes.baseAddress!.advanced(by: byteOffset), byteCount: byteCount)
        }
    }

    private func copySlice(
        _ source: Data,
        srcOffsetPoints: Int,
        dstOffsetPoints: Int,
        count: Int,
        stride: Int,
        into buffer: MTLBuffer?
    ) {
        guard let buffer, count > 0 else { return }
        let byteCount = count * stride
        let srcByteOffset = srcOffsetPoints * stride
        let dstByteOffset = dstOffsetPoints * stride
        precondition(srcByteOffset + byteCount <= source.count, "addBatches: source data too small for batch")
        precondition(dstByteOffset + byteCount <= buffer.length, "addBatches: dest buffer too small")
        source.withUnsafeBytes { srcRaw in
            let src = srcRaw.baseAddress!.advanced(by: srcByteOffset)
            buffer.contents().advanced(by: dstByteOffset).copyMemory(from: src, byteCount: byteCount)
        }
    }
}

// MARK: - Helpers

private func matrix_translation(_ t: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = SIMD4<Float>(t, 1)
    return m
}

extension ComputeRasteriserPointCloud {
    /// Free slot count. `addBatches` requires at least `n` of these.
    public var freeSlotCount: Int { freeSlots.count }
}
