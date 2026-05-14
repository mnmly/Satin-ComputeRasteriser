// Three-pass neighbour-average expansion. Valid pixels (α ≥ 0.5) pass
// through; empty pixels adopt the mean colour of their valid neighbours.
// Pattern from Magnopus's point-cloud rendering write-up.
#include "../Common.metal"

struct HoleFillUniforms {
    int2 screenSize;
};

kernel void holeFillUpdate(
    constant HoleFillUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    texture2d<float, access::read> inputTexture [[texture(ComputeTextureCustom0)]],
    texture2d<float, access::write> outputTexture [[texture(ComputeTextureCustom1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(uniforms.screenSize.x) || gid.y >= uint(uniforms.screenSize.y)) {
        return;
    }

    const float4 center = inputTexture.read(gid);
    if (center.a >= 0.5) {
        outputTexture.write(center, gid);
        return;
    }

    float3 sum = float3(0.0);
    float count = 0.0;
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            if (ox == 0 && oy == 0) continue;
            const int2 sample = int2(gid) + int2(ox, oy);
            if (sample.x < 0 || sample.x >= uniforms.screenSize.x || sample.y < 0 || sample.y >= uniforms.screenSize.y) {
                continue;
            }
            const float4 neighbour = inputTexture.read(uint2(sample));
            if (neighbour.a >= 0.5) {
                sum += neighbour.rgb;
                count += 1.0;
            }
        }
    }

    if (count > 0.0) {
        outputTexture.write(float4(sum / count, 1.0), gid);
    } else {
        outputTexture.write(center, gid);
    }
}
