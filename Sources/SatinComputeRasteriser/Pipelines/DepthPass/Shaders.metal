#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
// `tintAlphaIsCoverage` (with `applyTint`): translucent-defocus mode — defocused
// points skip the depth write (see depthPassUpdate) so they don't occlude.
struct DepthPassUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
    int lodDither;
    int applyDisplacement;
    int applyTint;
    int tintAlphaIsCoverage;
};

kernel void depthPassUpdate(
    constant DepthPassUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterBatch *batches [[buffer(ComputeBufferCustom0)]],
    device const uint *xyzLow [[buffer(ComputeBufferCustom1)]],
    device const uint *xyzMed [[buffer(ComputeBufferCustom2)]],
    device const uint *xyzHigh [[buffer(ComputeBufferCustom3)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom4)]],
    device RasterPixel *pixels [[buffer(ComputeBufferCustom5)]],
    device const uchar *levels [[buffer(ComputeBufferCustom6)]],
    device const VisibleBatch *visible [[buffer(ComputeBufferCustom7)]],
    device const float3 *displacements [[buffer(ComputeBufferCustom8)]],
    // Custom9 (colors) is unused by the depth pass; tints share Custom10 with the
    // colour pass so the buffer binds at one index across both processors.
    device const float4 *tints [[buffer(ComputeBufferCustom10)]],
    uint slot [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const VisibleBatch vb = visible[slot];
    const RasterBatch batch = batches[vb.batchIndex];
    const RasterFile file = files[batch.fileIndex];
    const int level = vb.level;
    const float lodThreshold = vb.lodThreshold;
    const uint pointsPerThread = (batch.numPoints + CR_THREADS_PER_GROUP - 1u) / CR_THREADS_PER_GROUP;

    for (uint i = 0; i < pointsPerThread; i++) {
        const uint localIndex = i * CR_THREADS_PER_GROUP + lid;
        if (localIndex >= batch.numPoints) {
            continue;
        }

        const uint pointIndex = batch.firstPoint + localIndex;
        const float pointLevel = float(uint(levels[pointIndex]) & 0x7u);
        const float dither = (uniforms.lodDither != 0) ? hashUnit(pointIndex) : 0.5;
        if (dither >= lodThreshold - pointLevel + 0.5) {
            continue;
        }
        // Translucent defocus: drop only the discard sentinel here. All other
        // points still write depth (used solely as the nearest cloud surface for
        // the scene composite). The colour pass does NOT depth-test in coverage
        // mode — it weighted-blends every point (order-independent), so there's no
        // hard occlusion edge at the focus boundary.
        if (uniforms.applyTint != 0 && uniforms.tintAlphaIsCoverage != 0) {
            if (tints[pointIndex].a < 0.0) { continue; }
        }
        float3 point = decodePoint(pointIndex, batch, xyzLow, xyzMed, xyzHigh, level);
        if (uniforms.applyDisplacement != 0) {
            point += displacements[pointIndex];
            // Cull sentinel: a NaN displacement drops the point from the depth
            // pass too, so it neither occludes nor draws (geometry behind shows).
            if (any(isnan(point))) { continue; }
        }
        float4 clip = file.transform * float4(point, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }

        const float3 ndc = clip.xyz / clip.w;
        if (any(ndc.xy < -1.0) || any(ndc.xy > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) {
            continue;
        }

        int2 pixelCoord = int2((ndc.xy * 0.5 + 0.5) * float2(uniforms.screenSize));
        if (pixelCoord.x < 0 || pixelCoord.x >= uniforms.screenSize.x || pixelCoord.y < 0 || pixelCoord.y >= uniforms.screenSize.y) {
            continue;
        }
        pixelCoord.y = uniforms.screenSize.y - 1 - pixelCoord.y;

        const uint depth = depthToUintReverseZ(ndc.z);
        const int radius = pointFootprintRadius(
            point, file,
            uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize,
            uniforms.pointSizeMode,
            uniforms.minimumPointSize, uniforms.maximumPointSize, uniforms.pointSizeScale
        );

        for (int oy = -radius; oy <= radius; oy++) {
            for (int ox = -radius; ox <= radius; ox++) {
                const int2 offset = int2(ox, oy);
                if (!insidePointFootprint(offset, radius)) {
                    continue;
                }

                const int2 target = pixelCoord + offset;
                if (target.x < 0 || target.x >= uniforms.screenSize.x || target.y < 0 || target.y >= uniforms.screenSize.y) {
                    continue;
                }

                const uint pixelIndex = uint(target.y * uniforms.screenSize.x + target.x);
                atomic_fetch_max_explicit((device atomic_uint *)&pixels[pixelIndex].depth, depth, memory_order_relaxed);
            }
        }
    }
}
