vertex VertexData computeRasteriserPostVertex(
    Vertex in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant VertexUniforms *vertexUniforms [[buffer(VertexBufferVertexUniforms)]]
) {
    VertexData out;
    out.position = float4(in.position, 1.0);
    out.texcoord = in.texcoord;
    return out;
}

// Legacy composite: blends the resolved cloud color over the target with no
// depth interaction (always on top). Used when `configuration.writesSceneDepth`
// is false, or on render passes that have no depth attachment.
fragment float4 computeRasteriserPostFragment(
    VertexData in [[stage_in]],
    texture2d<float> resultTexture [[texture(FragmentTextureCustom1)]]
) {
    constexpr sampler s(filter::linear, mip_filter::nearest);
    return resultTexture.sample(s, in.texcoord);
}

// Depth-aware composite: outputs the cloud's per-pixel reversed-Z NDC depth so
// regular Satin meshes (e.g. a selection bounding box, drawn in the scene pass
// before this composite) correctly inter-occlude with the cloud. Depth-test
// only — the cloud composites last among 3D, so it needn't write depth.
// Requires a depth attachment on the render pass. The depth texture stores 0
// for pixels with no cloud (far in reversed-Z), which we discard so the
// composite never touches background / mesh-only pixels.
struct ComputeRasteriserPostOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment ComputeRasteriserPostOut computeRasteriserPostDepthFragment(
    VertexData in [[stage_in]],
    texture2d<float> resultTexture [[texture(FragmentTextureCustom1)]],
    texture2d<float> depthTexture [[texture(FragmentTextureCustom2)]]
) {
    constexpr sampler colorSampler(filter::linear, mip_filter::nearest);
    constexpr sampler depthSampler(filter::nearest, mip_filter::nearest);
    const float d = depthTexture.sample(depthSampler, in.texcoord).r;
    if (d <= 0.0) {
        discard_fragment();
    }
    ComputeRasteriserPostOut out;
    out.color = resultTexture.sample(colorSampler, in.texcoord);
    out.depth = d;
    return out;
}
