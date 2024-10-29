/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "GPUInputModalSine.h"
#include "GPUUtil.h"

#define BLOCK_SIZE 256

__global__ void gpuModalSineKernel(int DIM, int n, double *s, double *dat,
				   double *P)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (idx < DIM) {
	s[idx] += dat[n] * P[idx];
    }
}

GPUInputModalSine::GPUInputModalSine(int DIM, double *dat, double *P,
				     int startTime, int duration)
{
    this->DIM = DIM;
    this->startTime = startTime;
    this->duration = duration;

    cudaMalloc((void **)&d_dat, duration * sizeof(double));
    cudaMalloc((void **)&d_P, DIM * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    cudaMemcpy(d_dat, dat, duration * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, DIM * sizeof(double), cudaMemcpyHostToDevice);
    if (checkCUDAError("cudaMemcpy")) {
	exit(1);
    }
}

GPUInputModalSine::~GPUInputModalSine()
{
    cudaFree(d_dat);
    cudaFree(d_P);
}

void GPUInputModalSine::runTimestep(int n, double *s)
{
    n -= startTime;
    if ((n >= 0) && (n < duration)) {
	int dg = (DIM / BLOCK_SIZE);
	int db = BLOCK_SIZE;
	if ((DIM % db) != 0) dg++;
	dim3 gridDim(dg);
	dim3 blockDim(db);

	gpuModalSineKernel<<<gridDim,blockDim>>>(DIM, n, s, d_dat, d_P);
    }
}
