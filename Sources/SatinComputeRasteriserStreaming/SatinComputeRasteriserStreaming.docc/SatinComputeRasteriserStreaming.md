#  ``SatinComputeRasteriserStreaming``

SwiftPDAL → `SatinComputeRasteriser` glue for out-of-core point clouds.

## Overview

This target is the bridge layer between a SwiftPDAL streaming source
(typically `CopcStreamingPointCloudSource`) and a
`ComputeRasteriserPointCloud` slot pool. It is shipped as a separate
library product so that consumers of the core renderer who bring their
own data source do not have to link PDAL or lazperf.

Per frame:

1. Submit the current camera view to the source.
2. Drain the driver's `(added, removed)` chunk delta.
3. Free evicted slots and upload new chunks.
4. Commit one batch-buffer flush at the end of the tick.

```swift
let source = try await CopcStreamingPointCloudSource.open(url, options: opts)
let adapter = StreamingAdapter(source: source, cloud: cloud)
adapter.setBudget(bytes: budgetBytes)

// each frame:
adapter.update(camera: camera, viewport: viewport)
```

See the `StreamingCOPCClouds` guide in the `SatinComputeRasteriser`
documentation for the end-to-end streaming workflow.

## Topics

### Streaming glue

- ``StreamingAdapter``
