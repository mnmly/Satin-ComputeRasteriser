# Converting PLY (or any LAS/LAZ) to COPC

Convert a PLY (or LAS/LAZ) file once with PDAL into a COPC the
streaming source can range-read.

## Why

`SatinComputeRasteriser` streams from [COPC](https://copc.io) (Cloud
Optimized Point Cloud) files via SwiftPDAL. PLY isn't streamable on
its own — convert it once with [PDAL](https://pdal.io) and ship the
COPC.

## The one-liner

```bash
pdal translate input.ply output.copc.laz \
  --writers.copc.forward=all
```

That's it for most cases. PDAL infers the reader from the extension,
maps `red`/`green`/`blue` to LAS RGB, and writes a single
`output.copc.laz` with the COPC octree VLR baked in.

For LAS/LAZ inputs (already lidar-shaped), the same command applies —
just swap the input extension. PDAL skips the format conversion and
re-tiles into the COPC hierarchy.

## When you need control: pipelines

```json
[
  "input.ply",
  {
    "type": "writers.copc",
    "filename": "output.copc.laz",
    "scale_x": 0.001,
    "scale_y": 0.001,
    "scale_z": 0.001,
    "a_srs": "EPSG:4326"
  }
]
```

Save as `pipeline.json` and run:

```bash
pdal pipeline pipeline.json
```

## Gotchas

### Pick the scale carefully

LAS stores positions as scaled `Int32`. The `scale_x/y/z` you choose
sets the smallest representable distance:

| Cloud kind | Suggested scale |
| --- | --- |
| Sub-meter prop / scanned object | `0.0001` (0.1 mm) |
| Building / room | `0.001` (1 mm) |
| City / outdoor lidar | `0.01` (1 cm) |

Wrong scale → visible quantization stair-stepping in the renderer.
Default to `0.001`; tighten if your dataset is small, loosen for
city-scale.

### RGB widths

PLY is usually 8-bit per channel; LAS RGB is 16-bit. PDAL upscales
(multiplies by 256) automatically. If your PLY *already* stored 16-bit
values into 8-bit fields (some exporters do this), the result will
look dim — divide back down by adding a `filters.assign`.

### Non-standard PLY attributes

By default, only XYZ + RGB carry over. To preserve a `confidence` or
`scalar_*` field as a LAS extra-byte:

```json
{ "type": "filters.ferry", "dimensions": "confidence => Confidence" }
```

The renderer doesn't currently consume extra bytes, but they survive
the round-trip if you ever need them downstream.

### CRS

PLY has no coordinate reference system. PDAL writes COPC without one
unless you supply `--writers.copc.a_srs="EPSG:XXXX"` (or `a_srs` in a
pipeline). The renderer doesn't care, but downstream geo tools
(QGIS, Cesium, ArcGIS) will complain. For non-georeferenced art /
scan / studio data, just leave it off.

### Binary PLY only, practically

ASCII PLY parses but is glacial on big files. Convert ASCII → binary
first if your source is ASCII:

```bash
# (PDAL re-emits as binary internally on read, but if you want a
# standalone binary PLY for other tools, MeshLab, plyfile, etc. work.)
```

## Verify

```bash
pdal info output.copc.laz --summary       # bounds, point count, dims
pdal info output.copc.laz --metadata | grep copc
```

If `pdal info` shows a `copc` block in the metadata, it's a real COPC
and any COPC reader (PDAL, CesiumJS, Potree v2, SwiftPDAL's
`CopcStreamingPointCloudSource`) can range-read it.

## Why not Potree?

Potree v1 (directory-of-thousands-of-files) and v2 (single binary +
hierarchy) are both read by SwiftPDAL only via PDAL's COPC bridge.
**COPC has won as the open streaming format for point clouds:**
single file, HTTP range-readable, OGC-standardized (2024), supported
by PDAL, QGIS, Cesium, Potree itself, and SwiftPDAL.

For the workflow this renderer targets — open a file in the example
app, stream chunks as the camera moves — convert once to COPC and
forget Potree.

## See also

- <doc:StreamingCOPCClouds> — wiring the resulting COPC into the renderer
- [COPC spec](https://copc.io)
- [PDAL docs](https://pdal.io)
