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
    const RasterFile file = files[batch.fileIndex];

    if (uniforms.enableFrustumCulling != 0) {
        const float3 wgMin = batchMin(batch);
        const float3 wgMax = batchMax(batch);
        if (!intersectsFrustum(file.transformFrustum, wgMin, wgMax)) {
            return;
        }
    }

    const int level = precisionLevel(batch, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    // Map projected-precision tier to a max LOD that should still render.
    // Tier 0 (very close) and 1 (moderately close) show all levels.
    // Tiers 2-4 progressively drop the finest levels.
    constexpr int lodForPrecision[5] = { 7, 7, 3, 2, 1 };
    const int tier = clamp(level, 0, 4);
    const int lodThreshold = max(0, lodForPrecision[tier] + uniforms.lodBias);
    const uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    visible[slot].batchIndex = gid;
    visible[slot].level = level;
    visible[slot].lodThreshold = lodThreshold;
    visible[slot].padding = 0u;
}
