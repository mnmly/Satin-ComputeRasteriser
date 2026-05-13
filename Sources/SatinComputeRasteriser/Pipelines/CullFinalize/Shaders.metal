#include "../Common.metal"

kernel void cullFinalizeUpdate(
    device const atomic_uint *counter [[buffer(ComputeBufferCustom0)]],
    device CRDispatchArgs *args [[buffer(ComputeBufferCustom1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    args[0].threadgroupsX = atomic_load_explicit(counter, memory_order_relaxed);
    args[0].threadgroupsY = 1u;
    args[0].threadgroupsZ = 1u;
}
