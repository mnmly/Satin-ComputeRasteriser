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
    uint padding3;
    uint padding4;
    uint padding5;
    uint padding6;
    uint padding7;
    uint padding8;
} RasterBatch;

typedef struct {
    float4x4 transform;
    float4x4 transformFrustum;
    float4x4 world;
    float4x4 padding;
} RasterFile;

typedef struct {
    uint depth;
    uint red;
    uint green;
    uint blue;
    uint count;
    uint3 padding;
} RasterPixel;

typedef struct {
    uint batchIndex;
    int level;
    int lodThreshold;
    uint padding;
} VisibleBatch;

typedef struct {
    uint threadgroupsX;
    uint threadgroupsY;
    uint threadgroupsZ;
} CRDispatchArgs;

#endif
