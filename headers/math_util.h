#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#if DEVICE == 0
#define SIGNATURE
#elif DEVICE == 1
#define SIGNATURE __global__

#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * This method returns the starting point and the number of elements on which the
 * calling thread should work with, in a parallel computation.
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
#endif

void matrixMatrixMultiply(double*, double*, double*, int, int, int);
SIGNATURE void add(double*, double*, int, int, double*);
SIGNATURE void sub(double*, double*, int, int, double*);
void rotate(double*, double*, int, double*);

#endif