#include <metal_stdlib>
#include "ComputeRasteriserTypes.h"
using namespace metal;

struct RasterPlane {
    float3 normal;
    float offset;
};

inline uint depthToUintReverseZ(float ndcZ) {
    return uint(saturate(ndcZ) * float(CR_MAX_DEPTH - 1u));
}

inline float uintToDepthReverseZ(uint depth) {
    return float(depth) / float(CR_MAX_DEPTH - 1u);
}

inline int pointFootprintRadius(
    float3 point,
    RasterFile file,
    float4x4 viewMatrix,
    float4x4 projectionMatrix,
    int2 screenSize,
    int pointSizeMode,
    float minimumPointSize,
    float maximumPointSize,
    float pointSizeScale
) {
    const float3 worldPoint = (file.world * float4(point, 1.0)).xyz;
    const float4 viewPos = viewMatrix * float4(worldPoint, 1.0);
    const float lo = max(min(minimumPointSize, maximumPointSize), 1.0);
    const float hi = max(max(minimumPointSize, maximumPointSize), lo);

    float pointSize;
    if (pointSizeMode == 1) {
        const float viewZ = max(-viewPos.z, 0.000001);
        const float focal = float(screenSize.y) * 0.5 * projectionMatrix[1][1];
        pointSize = 2.0 * pointSizeScale * focal / viewZ;
    } else {
        const float viewDistance = max(length(viewPos.xyz), 0.000001);
        pointSize = pointSizeScale / viewDistance;
    }
    if (!isfinite(pointSize)) {
        pointSize = lo;
    }
    pointSize = clamp(pointSize, lo, hi);
    return clamp(int(ceil(max(pointSize - 1.0, 0.0) * 0.5)), 0, 16);
}

inline bool insidePointFootprint(int2 offset, int radius) {
    if (radius <= 0) {
        return offset.x == 0 && offset.y == 0;
    }

    const float2 p = float2(offset);
    const float r = float(radius) + 0.5;
    return dot(p, p) <= r * r;
}

inline float3 batchMin(RasterBatch batch) {
    return float3(batch.minX, batch.minY, batch.minZ);
}

inline float3 batchMax(RasterBatch batch) {
    return float3(batch.maxX, batch.maxY, batch.maxZ);
}

inline float3 batchSize(RasterBatch batch) {
    return max(batchMax(batch) - batchMin(batch), float3(0.000001));
}

inline uint unpack10(uint encoded, uint shift) {
    return (encoded >> shift) & CR_MASK_10BIT;
}

inline float3 decodePoint(uint pointIndex, RasterBatch batch, device const uint *xyzLow, device const uint *xyzMed, device const uint *xyzHigh, int level) {
    const float3 wgMin = batchMin(batch);
    const float3 wgSize = batchSize(batch);

    if (level == 0) {
        const uint low = xyzLow[pointIndex];
        const uint med = xyzMed[pointIndex];
        const uint high = xyzHigh[pointIndex];
        const uint x = (unpack10(low, 0) << 20) | (unpack10(med, 0) << 10) | unpack10(high, 0);
        const uint y = (unpack10(low, 10) << 20) | (unpack10(med, 10) << 10) | unpack10(high, 10);
        const uint z = (unpack10(low, 20) << 20) | (unpack10(med, 20) << 10) | unpack10(high, 20);
        return float3(x, y, z) * (wgSize / CR_STEPS_30BIT) + wgMin;
    }

    if (level == 1) {
        const uint low = xyzLow[pointIndex];
        const uint med = xyzMed[pointIndex];
        const uint x = (unpack10(low, 0) << 10) | unpack10(med, 0);
        const uint y = (unpack10(low, 10) << 10) | unpack10(med, 10);
        const uint z = (unpack10(low, 20) << 10) | unpack10(med, 20);
        return float3(x, y, z) * (wgSize / CR_STEPS_20BIT) + wgMin;
    }

    const uint low = xyzLow[pointIndex];
    const uint x = unpack10(low, 0);
    const uint y = unpack10(low, 10);
    const uint z = unpack10(low, 20);
    return float3(x, y, z) * (wgSize / CR_STEPS_10BIT) + wgMin;
}

inline RasterPlane makePlane(float x, float y, float z, float w) {
    const float nLength = max(length(float3(x, y, z)), 0.000001);
    RasterPlane plane;
    plane.normal = float3(x, y, z) / nLength;
    plane.offset = w / nLength;
    return plane;
}

inline bool intersectsFrustum(float4x4 m, float3 wgMin, float3 wgMax) {
    RasterPlane planes[6] = {
        makePlane(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]),
        makePlane(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]),
        makePlane(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]),
        makePlane(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]),
        makePlane(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]),
        makePlane(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]),
    };

    for (int i = 0; i < 6; i++) {
        const RasterPlane plane = planes[i];
        const float3 p = float3(
            plane.normal.x > 0.0 ? wgMax.x : wgMin.x,
            plane.normal.y > 0.0 ? wgMax.y : wgMin.y,
            plane.normal.z > 0.0 ? wgMax.z : wgMin.z
        );
        if (dot(plane.normal, p) + plane.offset < 0.0) {
            return false;
        }
    }
    return true;
}

inline int precisionLevel(RasterBatch batch, RasterFile file, float4x4 viewMatrix, float4x4 projectionMatrix, int2 imageSize) {
    const float3 wgMin = batchMin(batch);
    const float3 wgMax = batchMax(batch);
    const float3 wgCenter = (wgMin + wgMax) * 0.5;
    const float wgRadius = distance(wgMin, wgMax);

    const float4 viewCenter = viewMatrix * file.world * float4(wgCenter, 1.0);
    const float4 viewEdge = viewCenter + float4(wgRadius, 0.0, 0.0, 0.0);
    float4 projCenter = projectionMatrix * viewCenter;
    float4 projEdge = projectionMatrix * viewEdge;

    projCenter.xy /= max(abs(projCenter.w), 0.000001);
    projEdge.xy /= max(abs(projEdge.w), 0.000001);

    const float2 screenCenter = float2(imageSize) * (projCenter.xy + 1.0) * 0.5;
    const float2 screenEdge = float2(imageSize) * (projEdge.xy + 1.0) * 0.5;
    const float pixelSize = distance(screenEdge, screenCenter);

    if (pixelSize < 100.0) { return 4; }
    if (pixelSize < 200.0) { return 3; }
    if (pixelSize < 500.0) { return 2; }
    if (pixelSize < 10000.0) { return 1; }
    return 0;
}
