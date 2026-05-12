#include "../Common.metal"

struct ClearWinnerUniforms {
    int pixelCount;
};

kernel void clearWinnerUpdate(
    constant ClearWinnerUniforms &uniforms [[buffer(ComputeBufferUniforms)]],
    device uint *depths [[buffer(ComputeBufferCustom0)]],
    device uint *indices [[buffer(ComputeBufferCustom1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(uniforms.pixelCount)) {
        return;
    }

    depths[gid] = 0u;
    indices[gid] = 0xffffffffu;
}
