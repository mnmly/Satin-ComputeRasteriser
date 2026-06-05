#include "../Common.metal"

// ⚠️ No inline `//` comments INSIDE this struct — Satin's uniform parser drops
// the field after a comment (set() silently no-ops). Document fields above it.
// `coverageEnabled`: resolve the weighted-blended-OIT accumulation (translucent
// defocus) instead of the count-average — see resolveUpdate.
struct ResolveUniforms {
    int2 screenSize;
    float4 backgroundColor;
    int coverageEnabled;
};

kernel void resolveUpdate(
    constant ResolveUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device const RasterPixel *pixels [[buffer(ComputeBufferCustom0)]],
    texture2d<float, access::write> outputTexture [[texture(ComputeTextureCustom0)]],
    texture2d<float, access::write> depthTexture [[texture(ComputeTextureCustom1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const uint pixelIndex = gid.y * uint(uniforms.screenSize.x) + gid.x;
    const RasterPixel pixel = pixels[pixelIndex];

    if (uniforms.coverageEnabled != 0) {
        // Weighted-blended OIT resolve (fixed-point, S = 4096):
        //   red/green/blue = Σ(colour·α·S)   count = Σ(α·S)
        // Colour is the α-weighted average (S cancels); the pixel alpha is the
        // additive revealage 1 - e^(-Σα), so focus→defocus composites as a smooth,
        // order-independent blend with no hard occlusion edge.
        const uint aSumFixed = pixel.count;
        if (aSumFixed == 0u) {
            outputTexture.write(float4(uniforms.backgroundColor.rgb, 0.0), gid);
            depthTexture.write(float4(0.0), gid);
            return;
        }
        const float3 rgb  = float3(pixel.red, pixel.green, pixel.blue) / float(aSumFixed);
        const float sumA  = float(aSumFixed) * (1.0 / 4096.0);    // Σα
        const float alpha = saturate(1.0 - exp(-sumA));
        outputTexture.write(float4(rgb, alpha), gid);
        depthTexture.write(float4(uintToDepthReverseZ(pixel.depth)), gid);
        return;
    }

    if (pixel.count == 0u) {
        outputTexture.write(float4(uniforms.backgroundColor.rgb, 0.0), gid);
        depthTexture.write(float4(0.0), gid);   // 0 = no cloud (far in reversed-Z)
        return;
    }

    const float invCount = 1.0 / float(pixel.count);
    const float3 rgb = float3(pixel.red, pixel.green, pixel.blue) * invCount / 255.0;
    outputTexture.write(float4(rgb, 1.0), gid);
    depthTexture.write(float4(uintToDepthReverseZ(pixel.depth)), gid);
}

