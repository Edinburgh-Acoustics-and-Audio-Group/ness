/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * CUDA utility functions
 */
#ifndef _GPU_UTIL_H_
#define _GPU_UTIL_H_

extern "C" {
#include "csrmatrix.h"
};

#include <string>
using namespace std;

/* the pointers are to device memory */
typedef struct GpuMatrix_s {
    int nrow;
    int ncol;
    int valuesPerRow;
    int *colIndex;
    double *values;
} GpuMatrix_t;

GpuMatrix_t *csrToGpuMatrix(CSRmatrix *csr);
void freeGpuMatrix(GpuMatrix_t *mat);
void gpuMatrixMultiply(GpuMatrix_t *mat, double *invec, double *outvec);

bool checkCUDAError(string msg);

void cudaMemcpyH2D(void *dest, void *src, int size);

void cudaMemcpyD2H(void *dest, void *src, int size);

void *cudaMallocNess(int size);
void cudaFreeNess(void *ptr);
void cudaMemsetNess(void *ptr, int val, int size);

#endif
