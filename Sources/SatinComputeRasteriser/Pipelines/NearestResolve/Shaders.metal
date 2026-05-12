#include "../Common.metal"

struct NearestResolveUniforms {
    int2 screenSize;
    float4 backgroundColor;
};

kernel void nearestResolveUpdate(
    constant NearestResolveUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const uint *depths [[buffer(ComputeBufferCustom0)]],
    device const uint *indices [[buffer(ComputeBufferCustom1)]],
    device const uint *colors [[buffer(ComputeBufferCustom2)]],
    texture2d<float, access::write> outputTexture [[texture(ComputeTextureCustom0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const uint pixelIndex = gid.y * uint(uniforms.screenSize.x) + gid.x;
    const uint depth = depths[pixelIndex];
    const uint pointIndex = indices[pixelIndex];
    if (depth == 0u || pointIndex == 0xffffffffu) {
        outputTexture.write(uniforms.backgroundColor, gid);
        return;
    }

    const uint rgba = colors[pointIndex];
    const float3 rgb = float3(
        float((rgba >> 0u) & 0xffu),
        float((rgba >> 8u) & 0xffu),
        float((rgba >> 16u) & 0xffu)
    ) / 255.0;
    outputTexture.write(float4(rgb, 1.0), gid);
}
