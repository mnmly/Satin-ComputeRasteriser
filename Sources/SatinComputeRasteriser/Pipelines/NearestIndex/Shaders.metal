#include "../Common.metal"

struct NearestIndexUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int enableFrustumCulling;
};

kernel void nearestIndexUpdate(
    constant NearestIndexUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterBatch *batches [[buffer(ComputeBufferCustom0)]],
    device const uint *xyzLow [[buffer(ComputeBufferCustom1)]],
    device const uint *xyzMed [[buffer(ComputeBufferCustom2)]],
    device const uint *xyzHigh [[buffer(ComputeBufferCustom3)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom4)]],
    device const uint *depths [[buffer(ComputeBufferCustom5)]],
    device atomic_uint *indices [[buffer(ComputeBufferCustom6)]],
    uint batchIndex [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const RasterBatch batch = batches[batchIndex];
    const RasterFile file = files[batch.fileIndex];
    const float3 wgMin = batchMin(batch);
    const float3 wgMax = batchMax(batch);

    if (uniforms.enableFrustumCulling != 0 && !intersectsFrustum(file.transformFrustum, wgMin, wgMax)) {
        return;
    }

    const int level = precisionLevel(batch, file, uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    const uint pointsPerThread = (batch.numPoints + CR_THREADS_PER_GROUP - 1u) / CR_THREADS_PER_GROUP;

    for (uint i = 0; i < pointsPerThread; i++) {
        const uint localIndex = i * CR_THREADS_PER_GROUP + lid;
        if (localIndex >= batch.numPoints) {
            continue;
        }

        const uint pointIndex = batch.firstPoint + localIndex;
        const float3 point = decodePoint(pointIndex, batch, xyzLow, xyzMed, xyzHigh, level);
        float4 clip = file.transform * float4(point, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }

        const float3 ndc = clip.xyz / clip.w;
        if (any(ndc.xy < -1.0) || any(ndc.xy > 1.0) || ndc.z <= 0.0 || ndc.z > 1.0) {
            continue;
        }

        int2 pixelCoord = int2((ndc.xy * 0.5 + 0.5) * float2(uniforms.screenSize));
        if (pixelCoord.x < 0 || pixelCoord.x >= uniforms.screenSize.x || pixelCoord.y < 0 || pixelCoord.y >= uniforms.screenSize.y) {
            continue;
        }
        pixelCoord.y = uniforms.screenSize.y - 1 - pixelCoord.y;

        const uint pixelIndex = uint(pixelCoord.y * uniforms.screenSize.x + pixelCoord.x);
        const uint depth = depthToUintReverseZ(ndc.z);
        if (depth == depths[pixelIndex]) {
            atomic_fetch_min_explicit(&indices[pixelIndex], pointIndex, memory_order_relaxed);
        }
    }
}
