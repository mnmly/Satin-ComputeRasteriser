# Cross-package layout invariants

How ``RasterBatch`` matches SwiftPDAL's `StreamingRasterBatch`, and why
that matters.

## Overview

Streaming sources (notably `SwiftPDAL.CopcStreamingPointCloudSource`)
hand the renderer a `ResidentChunk` whose `batches: [StreamingRasterBatch]`
gets memcpy'd straight into this package's `batchesBuffer`. The two
structs are intentionally laid out byte-for-byte the same.

### The contract

| Offset | Bytes | Field | Notes |
| -----: | ----: | ----- | ----- |
| 0  | 4  | `state`       | `0` = empty/evicting, `1` = resident |
| 4  | 4  | `minX`        | batch AABB min, in cloud-local coordinates |
| 8  | 4  | `minY`        | |
| 12 | 4  | `minZ`        | |
| 16 | 4  | `maxX`        | batch AABB max |
| 20 | 4  | `maxY`        | |
| 24 | 4  | `maxZ`        | |
| 28 | 4  | `numPoints`   | UInt32, ≤ `pointsPerBatch` (10240) |
| 32 | 4  | `firstPoint`  | UInt32, slot-index × `pointsPerBatch` |
| 36 | 4  | `fileIndex`   | UInt32, index into `RasterFile` table |
| 40 | 24 | padding 3..8  | unused; keeps stride at 64 for Metal |

Total stride: **64 bytes**.

Position buffers (`xyzLow`/`Med`/`High`), `colors`, and `levels` use the
slot range `firstPoint ..< firstPoint + numPoints`.

### Verification

A test in `Tests/SatinComputeRasteriserTests/LayoutTests.swift` pins each
field offset. If you change the struct shape, that test fails first and
the matching change must land in SwiftPDAL's `StreamingRasterBatch`
before the streaming path will work again.

### Why memcpy

Decoding chunks runs on a background actor; the result is bit-identical
to what the GPU consumes. Letting the source layer pre-pack means the
render thread does one `MTLBlitCommandEncoder.copy(from:..to:..)` per
upload instead of touching each batch.
