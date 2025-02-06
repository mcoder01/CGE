#ifndef RENDER_H
#define RENDER_H

#include "cuda_util.h"

#if DEVICE == 0
#define SIGNATURE
#elif DEVICE == 1
#define SIGNATURE __global__
#endif

typedef struct {
    double position[3];
    double texCoord[2];
    double normal[3];
    double projection[2];
} Vertex;

typedef struct {
    Vertex* v1;
    Vertex* v2;
} Side;

/**
 * Performs a linear interpolation
 */
inline __host__ __device__ double lerp(double start, double end, double t) {
    return t*(end-start)+start;
}

/**
 * Computes the linear interpolation between two vectors and store the result into array `output`.
 */
inline __host__ __device__ void lerpVector(double* start, double* end, double t, int dim, double* output) {
    for (int i = 0; i < dim; i++)
        output[i] = lerp(start[i], end[i], t);
}

/**
 * Computes the linear interpolation between two vertices, by interpolating the vectors that constitute them.
 */
inline __host__ __device__ Vertex lerpVertex(Vertex* start, Vertex* end, double t) {
    Vertex result;
    lerpVector(start->position, end->position, t, 3, result.position);
    lerpVector(start->texCoord, end->texCoord, t, 2, result.texCoord);
    lerpVector(start->normal, end->normal, t, 3, result.normal);
    lerpVector(start->projection, end->projection, t, 2, result.projection);
    return result;
}

__global__ void initScreen(DeviceScreen);
SIGNATURE void raster(Vertex*, int, Texture, DeviceScreen, SDL_Rect);

#endif