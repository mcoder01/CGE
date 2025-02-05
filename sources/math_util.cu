#include "../headers/math_util.h"

#if DEVICE == 0

/**
 * Computes a matrix-to-matrix multiplication (rows by columns) between matrix A (mxk) 
 * and matrix B (kxn). The result is stored into the matrix C (mxn).
 */
void matrixMatrixMultiply(double* A, double* B, double* C, int m, int k, int n) {
    double result[m*n];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            result[i*n+j] = 0;
            for (int h = 0; h < k; h++)
                result[i*n+j] += A[i*k+h]*B[h*n+j];
        }

    memcpy(C, result, m*n*sizeof(double));
}

/**
 * Adds the vector `v` to the rows of the matrix `A` and stores the result in the corresponding matrix.
 */
void add(double* A, double* v, int m, int n, double* result) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int idx = i*n+j;
            result[idx] = A[idx]+v[j];
        }
}

/**
 * Subtracts the vector `v` from the rows of the matrix `A` and stores the result in the corresponding matrix.
 */
void sub(double* A, double* v, int m, int n, double* result) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int idx = i*n+j;
            result[idx] = A[idx]-v[j];
        }
}

/**
 * Generates the Euler rotation matrix of the given angle and for the given axis.
 * The result is stored into `matrix`.
 */
void rotationMatrix(double* angles, int axis, double* matrix) {
    double sinAngle = sin(angles[axis]), cosAngle = cos(angles[axis]);
    memset(matrix, 0, 9*sizeof(double));
    if (axis == 0) {
        matrix[0] = 1;
        matrix[4] = matrix[8] = cosAngle;
        matrix[5] = -sinAngle;
        matrix[7] = sinAngle;
    } else {
        matrix[0] = matrix[8] = cosAngle;
        matrix[2] = sinAngle;
        matrix[4] = 1;
        matrix[6] = -sinAngle;
    }
}

/**
 * Rotates the given point by the given angles for X and Y axes.
 * The rotation is firstly performed on the Y-axis and then on the X-axis.
 */
void rotate(double* points, double* angles, int num_points, double* output) {
    double rotX[9], rotY[9];
    rotationMatrix(angles, 1, rotY);
    matrixMatrixMultiply(points, rotY, output, num_points, 3, 3);
    rotationMatrix(angles, 0, rotX);
    matrixMatrixMultiply(output, rotX, output, num_points, 3, 3);
}

#elif DEVICE == 1
/**
 * Computes a matrix-to-matrix multiplication (rows by columns) between the matrix A (mxk) 
 * and the matrix B (kxn) through the cuBLAS library. The result is stored into the matrix C (mxn).
 */
void matrixMatrixMultiply(double* A, double* B, double* C, int m, int k, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1, beta = 0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    cublasDestroy(handle);
}

/**
 * Parallelize the sum of the rows of `A` with the vector `v`.
 */
__global__ void add(double* A, double* v, int m, int n, double* result) {
    int start, size;
    distribute(m, 1, &start, &size);
    for (int i = start; i < start+size; i++)
        if (i < m)
            for (int j = 0; j < n; j++) {
                int idx = i*n+j;
                result[idx] = A[idx]+v[j];
            }
}

/**
 * Parallelize the subtraction between the rows of `A` and the vector `v`.
 */
__global__ void sub(double* A, double* v, int m, int n, double* result) {
    int start, size;
    distribute(m, 1, &start, &size);
    for (int i = start; i < start+size; i++)
        if (i < m)
            for (int j = 0; j < n; j++) {
                int idx = i*n+j;
                result[idx] = A[idx]-v[j];
            }
}

/**
 * Generates the Euler rotation matrix of the given angle and for the given axis.
 * The result is stored into `matrix`.
 */
__global__ void rotationMatrix(double* angles, int axis, double* matrix) {
    double sinAngle = sin(angles[axis]), cosAngle = cos(angles[axis]);
    memset(matrix, 0, 9*sizeof(double));
    if (axis == 0) {
        matrix[0] = 1;
        matrix[4] = matrix[8] = cosAngle;
        matrix[5] = -sinAngle;
        matrix[7] = sinAngle;
    } else {
        matrix[0] = matrix[8] = cosAngle;
        matrix[2] = sinAngle;
        matrix[4] = 1;
        matrix[6] = -sinAngle;
    }
}

/**
 * Rotates the given point by the given angles for X and Y axes.
 * The rotation is firstly performed on the Y-axis and then on the X-axis.
 */
void rotate(double* d_points, double* angles, int num_points, double* output) {
    double *rotX, *rotY;
    cudaMalloc((void**) &rotX, 9*sizeof(double));
    cudaMalloc((void**) &rotY, 9*sizeof(double));

    rotationMatrix<<<1,1>>>(angles, 1, rotY);
    matrixMatrixMultiply(rotY, d_points, output, 3, 3, num_points);
    rotationMatrix<<<1,1>>>(angles, 0, rotX);
    matrixMatrixMultiply(rotX, output, output, 3, 3, num_points);

    cudaFree(rotX);
    cudaFree(rotY);
}

#endif