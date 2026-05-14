// Batch culling + precision-level selection, run once per frame per cloud.
// Output feeds an indirect dispatch so depth/color/nearest passes only launch
// threadgroups for surviving batches. Pattern mirrors upstream
// compute_rasterizer (Schuetz et al.) and the Magnopus point-cloud write-up.
#include "../Common.metal"

struct CullUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int batchCount;
    int enableFrustumCulling;
    int lodBias;
    int enableCLOD;
};

kernel void cullUpdate(
    constant CullUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterBatch *batches [[buffer(ComputeBufferCustom0)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom1)]],
    device VisibleBatch *visible [[buffer(ComputeBufferCustom2)]],
    device atomic_uint *counter [[buffer(ComputeBufferCustom3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(uniforms.batchCount)) {
        return;
    }

    const RasterBatch batch = batches[gid];
    // Residency gate: streaming sources mark non-resident slots with state == 0.
    if (batch.state == 0) {
        return;
    }
    const RasterFile file = files[batch.fileIndex];

    if (uniforms.enableFrustumCulling != 0) {
        const float3 wgMin = batchMin(batch);
        const float3 wgMax = batchMax(batch);
        if (!intersectsFrustum(file.transformFrustum, wgMin, wgMax)) {
            return;
        }
    }

    const float pixelSize = pixelSizeOnScreen(batch, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    int level;
    if (pixelSize < 100.0) level = 4;
    else if (pixelSize < 200.0) level = 3;
    else if (pixelSize < 500.0) level = 2;
    else if (pixelSize < 10000.0) level = 1;
    else level = 0;

    // CLOD off: sentinel that's always ≥ dither(1.0) + pointLevel(7) + 0.5,
    // so the depth/color/nearest passes' `dither >= lodThreshold - pointLevel + 0.5`
    // test never culls.
    const float lodThreshold = (uniforms.enableCLOD != 0)
        ? max(0.0, lodThresholdFromPixelSize(pixelSize, uniforms.lodBias))
        : 99.0;
    const uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    visible[slot].batchIndex = gid;
    visible[slot].level = level;
    visible[slot].lodThreshold = lodThreshold;
    visible[slot].padding = 0u;
}
