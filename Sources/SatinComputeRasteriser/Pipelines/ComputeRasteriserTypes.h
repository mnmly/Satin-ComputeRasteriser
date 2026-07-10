#ifndef ComputeRasteriserTypes_h
#define ComputeRasteriserTypes_h

#define CR_THREADS_PER_GROUP 128
#define CR_STEPS_30BIT 1073741824.0f
#define CR_STEPS_20BIT 1048576.0f
#define CR_STEPS_10BIT 1024.0f
#define CR_MASK_10BIT 1023u
#define CR_MAX_DEPTH 0xffffffffu

typedef struct {
    int state;
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
    uint numPoints;
    uint firstPoint;
    uint fileIndex;
    // Cumulative LOD level counts, two uint16 per word (low level in the low
    // half): lodCum01 = cum0 | cum1<<16 … lodCum67 = cum6 | cum7<<16, where
    // cum[L] = points in the batch with level <= L. cum7 == numPoints for a
    // bucketed batch, so lodCum67 == 0 is the legacy sentinel (draw the full
    // numPoints range). Mirrors RasterBatch.padding3..6 on the Swift side.
    uint lodCum01;
    uint lodCum23;
    uint lodCum45;
    uint lodCum67;
    uint padding7;
    uint padding8;
} RasterBatch;

typedef struct {
    float4x4 transform;
    float4x4 transformFrustum;
    float4x4 world;
    float4x4 prevTransform;
} RasterFile;

typedef struct {
    uint depth;
    uint red;
    uint green;
    uint blue;
    uint count;
    uint weight;   // Σ(coverage·255) for translucent-defocus accumulation (else 0)
    uint2 padding;
} RasterPixel;

typedef struct {
    uint batchIndex;
    int level;
    float lodThreshold;
    uint activePoints;   // LOD survivor prefix length computed by cullUpdate
} VisibleBatch;

typedef struct {
    uint threadgroupsX;
    uint threadgroupsY;
    uint threadgroupsZ;
} CRDispatchArgs;

#endif
