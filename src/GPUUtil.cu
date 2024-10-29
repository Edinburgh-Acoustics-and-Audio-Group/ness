/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "GPUUtil.h"
#include "Logger.h"

#include <cuda_runtime.h>

#include <cstdlib>
using namespace std;

// converts a CSR matrix on the CPU to a simpler representation with constant row length
// that is more suitable for the GPU
GpuMatrix_t *csrToGpuMatrix(CSRmatrix *csr)
{
    GpuMatrix_t *result = new GpuMatrix_t;
    result->nrow = csr->nrow;
    result->ncol = csr->ncol;
    
    // count up how many items are in each row (max)
    int maxlen;
    for (int i = 0; i < csr->nrow; i++) {
	int rowlen = csr->rowStart[i+1] - csr->rowStart[i];
	if (rowlen > maxlen) maxlen = rowlen;
    }

    result->valuesPerRow = maxlen;
    logMessage(1, "Initialising %dx%d GPU matrix with %d values per row", result->nrow,
	       result->ncol, result->valuesPerRow);

    // allocate temporary buffers on the CPU
    int *tci = new int[result->nrow * result->valuesPerRow];
    double *tv = new double[result->nrow * result->valuesPerRow];

    // convert matrix data to fixed row length format
    for (int i = 0; i < result->nrow; i++) {
	int k = 0;
	for (int j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    tci[(i * result->valuesPerRow) + k] = csr->colIndex[j];
	    tv[(i * result->valuesPerRow) + k] = csr->values[j];
	    k++;
	}
	while (k < result->valuesPerRow) {
	    tci[(i * result->valuesPerRow) + k] = 0;
	    tv[(i * result->valuesPerRow) + k] = 0.0;
	    k++;
	}
    }

    // allocate GPU storage
    result->colIndex = (int*)cudaMallocNess(result->nrow * result->valuesPerRow * sizeof(int));
    result->values = (double*)cudaMallocNess(result->nrow * result->valuesPerRow * sizeof(double));

    // copy to GPU
    cudaMemcpyH2D(result->colIndex, tci, result->nrow * result->valuesPerRow * sizeof(int));
    cudaMemcpyH2D(result->values, tv, result->nrow * result->valuesPerRow * sizeof(double));
    
    delete[] tci;
    delete[] tv;
    return result;
}

void freeGpuMatrix(GpuMatrix_t *mat)
{
    cudaFree(mat->colIndex);
    cudaFree(mat->values);
    delete mat;
}

__global__ void gpuMatrixMultiplyKernel(int nrow, int valuesPerRow, int *colIndex, double *values,
					double *invec, double *outvec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < nrow) {
	double sum = 0.0;
	for (int j = 0; j < valuesPerRow; j++) {
	    sum += values[(i * valuesPerRow) + j] * invec[colIndex[(i * valuesPerRow) + j]];
	}
	outvec[i] = sum;
    }
}

// perform a matrix-by-vector multiply on the GPU
void gpuMatrixMultiply(GpuMatrix_t *mat, double *invec, double *outvec)
{
    int BLOCK_SIZE = 256;
    int gridSize = mat->nrow / BLOCK_SIZE;
    if ((mat->nrow % BLOCK_SIZE) != 0) gridSize++;
    dim3 gridDim(gridSize);
    dim3 blockDim(BLOCK_SIZE);
    gpuMatrixMultiplyKernel<<<gridDim, blockDim>>>(mat->nrow, mat->valuesPerRow,
						   mat->colIndex, mat->values,
						   invec, outvec);
}

bool checkCUDAError(string msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
	logMessage(5, "CUDA error: %s: %s", msg.c_str(), cudaGetErrorString(err));
	return true;
    }
    return false;
}

// wrap cudaMemcpy so it can be called from .cpp files without include path problems
void cudaMemcpyH2D(void *dest, void *src, int size)
{
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void cudaMemcpyD2H(void *dest, void *src, int size)
{
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

// similarly cudaMalloc and cudaFree
void *cudaMallocNess(int size)
{
    void *result;
    cudaMalloc(&result, size);
    checkCUDAError("cudaMalloc");
    return result;
}

void cudaFreeNess(void *ptr)
{
    cudaFree(ptr);
}

void cudaMemsetNess(void *ptr, int val, int size)
{
    cudaMemset(ptr, val, size);
}

