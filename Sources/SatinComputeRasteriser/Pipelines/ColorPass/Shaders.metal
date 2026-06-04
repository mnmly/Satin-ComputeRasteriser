#include "../Common.metal"

struct ColorPassUniforms {
    int2 screenSize;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float depthTolerance;
    int colorizeChunks;
    int colorizeOverdraw;
    int pointSizeMode;
    float minimumPointSize;
    float maximumPointSize;
    float pointSizeScale;
    int lodDither;
    int applyDisplacement;
    int applyTint;
};

kernel void colorPassUpdate(
    constant ColorPassUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterBatch *batches [[buffer(ComputeBufferCustom0)]],
    device const uint *xyzLow [[buffer(ComputeBufferCustom1)]],
    device const uint *xyzMed [[buffer(ComputeBufferCustom2)]],
    device const uint *xyzHigh [[buffer(ComputeBufferCustom3)]],
    device const RasterFile *files [[buffer(ComputeBufferCustom4)]],
    device RasterPixel *pixels [[buffer(ComputeBufferCustom5)]],
    device const uchar *levels [[buffer(ComputeBufferCustom6)]],
    device const VisibleBatch *visible [[buffer(ComputeBufferCustom7)]],
    device const float3 *displacements [[buffer(ComputeBufferCustom8)]],
    device const uint *colors [[buffer(ComputeBufferCustom9)]],
    device const float4 *tints [[buffer(ComputeBufferCustom10)]],
    uint slot [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const VisibleBatch vb = visible[slot];
    const uint batchIndex = vb.batchIndex;
    const RasterBatch batch = batches[batchIndex];
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
        float3 point = decodePoint(pointIndex, batch, xyzLow, xyzMed, xyzHigh, level);
        if (uniforms.applyDisplacement != 0) {
            point += displacements[pointIndex];
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

        uint color = colors[pointIndex];
        if (uniforms.colorizeChunks != 0) {
            color = batchIndex * 1234567u;
        } else if (uniforms.colorizeOverdraw != 0) {
            color = 0x00010101u;
        }

        uint r = color & 0xffu;
        uint g = (color >> 8) & 0xffu;
        uint b = (color >> 16) & 0xffu;

        if (uniforms.applyTint != 0) {
            const float4 tint = tints[pointIndex];
            // Discard sentinel: a negative tint alpha drops this point entirely
            // (no colour write, count not incremented). A pixel covered only by
            // discarded points resolves to count==0 → background alpha 0, i.e.
            // the point becomes fully transparent.
            if (tint.a < 0.0) { continue; }
            const float w = saturate(tint.a);
            const float3 orig = float3(float(r), float(g), float(b)) * (1.0 / 255.0);
            const float3 mixed = mix(orig, saturate(tint.rgb), w);
            r = uint(saturate(mixed.x) * 255.0 + 0.5);
            g = uint(saturate(mixed.y) * 255.0 + 0.5);
            b = uint(saturate(mixed.z) * 255.0 + 0.5);
        }
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
                const uint closestDepthUint = pixels[pixelIndex].depth;
                if (closestDepthUint == 0u) {
                    continue;
                }

                const float closestDepth = uintToDepthReverseZ(closestDepthUint);
                const bool visible = ndc.z >= closestDepth * (1.0 - uniforms.depthTolerance);
                if (!visible) {
                    continue;
                }

                atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].red, r, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].green, g, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].blue, b, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_uint *)&pixels[pixelIndex].count, 1u, memory_order_relaxed);
            }
        }
    }
}
