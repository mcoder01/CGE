#include "../headers/world.h"
#include "../headers/math_util.h"
#include "../headers/render.h"

/**
 * Performs the dot product between two vectors.
 */
__host__ __device__ double dotProduct(double* a, double* b, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++)
        sum += a[i]*b[i];
    return sum;
}

/**
 * Performs the normalization of the given vector in-place.
 */
__host__ __device__ void normalize(double* v, int len) {
    double mag = sqrt(dotProduct(v, v, len));
    for (int i = 0; i < len; i++)
        v[i] /= mag;
}

/**
 * Check if the given vertex is in the frustum
 */
__host__ __device__ bool isInFrustum(Vertex v, Plane plane) {
    double pd = dotProduct(plane.position, plane.normal, 3);
    double vd = dotProduct(v.position, plane.normal, 3);
    return vd-pd >= 0;
}

/**
 * Returns the factor which describes the intersection point between 
 * a plane and a segment represented by its ends (vectors `a` and `b`).
 * Input vectors must be of three dimensions.
 */
__host__ __device__ double frustumIntersection(double* a, double* b, Plane plane) {
    double pd = dotProduct(plane.position, plane.normal, 3);
    double ad = dotProduct(a, plane.normal, 3);
    double bd = dotProduct(b, plane.normal, 3);
    return (pd-ad)/(bd-ad);
}

World::~World() {
    printf("Average points viewing time: %f\n", viewPointTime/frames);
    printf("Average normal computation time: %f\n", normalComputationTime/frames);
    printf("Average face decomposition time: %f\n", faceDecompositionTime/frames);
    printf("Average face clipping time: %f\n", faceClippingTime/frames);
    printf("Average projection time: %f\n", projectionTime/frames);
    printf("Average area computation time: %f\n", viewPointTime/frames);
    printf("Average raster time: %f\n", rasterTime/frames);
    printf("Average frame time: %f\n", frameTime/frames);
}

void World::addMesh(Mesh mesh) {
    objects.push_back(mesh);
}

void World::tick(double deltaTime) {
    objects[0].rot[1] += 0.1*deltaTime;
}

#if DEVICE == 0

#include "cpu/world.cpp"

#elif DEVICE == 1

#include "gpu/world.cu"

#endif