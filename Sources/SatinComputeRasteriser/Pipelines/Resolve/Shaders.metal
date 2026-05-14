#include "../Common.metal"

struct ResolveUniforms {
    int2 screenSize;
    float4 backgroundColor;
};

kernel void resolveUpdate(
    constant ResolveUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterPixel *pixels [[buffer(ComputeBufferCustom0)]],
    texture2d<float, access::write> outputTexture [[texture(ComputeTextureCustom0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const uint pixelIndex = gid.y * uint(uniforms.screenSize.x) + gid.x;
    const RasterPixel pixel = pixels[pixelIndex];
    if (pixel.count == 0u) {
        outputTexture.write(float4(uniforms.backgroundColor.rgb, 0.0), gid);
        return;
    }

    const float invCount = 1.0 / float(pixel.count);
    const float3 rgb = float3(pixel.red, pixel.green, pixel.blue) * invCount / 255.0;
    outputTexture.write(float4(rgb, 1.0), gid);
}

