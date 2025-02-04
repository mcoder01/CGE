#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * This method returns the starting point and the number of elements on which the
 * calling thread must work with in a parallel computation.
 */
inline __device__ void distribute(int work_size, int skip, int* start, int* nloc) {
    int threads = gridDim.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    int scaled_size = work_size/skip;
    int loc = scaled_size/threads;
    int carry = scaled_size%threads;
    if (index < carry) {
        *nloc = (loc+1)*skip;
        *start = *nloc*index;
    } else {
        *nloc = loc*skip;
        *start = *nloc*index + carry*skip;
    }
}

void matrixMatrixMultiply(double*, double*, double*, int, int, int);
__global__ void add(double*, double*, int, int, double*);
__global__ void sub(double*, double*, int, int, double*);
void rotate(double*, double*, int, double*);

#endif