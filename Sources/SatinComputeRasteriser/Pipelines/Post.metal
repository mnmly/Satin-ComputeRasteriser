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

fragment float4 computeRasteriserPostFragment(
    VertexData in [[stage_in]],
    texture2d<float> resultTexture [[texture(FragmentTextureCustom1)]]
) {
    constexpr sampler s(filter::linear, mip_filter::nearest);
    return resultTexture.sample(s, in.texcoord);
}

