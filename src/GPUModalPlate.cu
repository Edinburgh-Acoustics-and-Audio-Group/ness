/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "GPUModalPlate.h"
#include "GPUUtil.h"
#include "Logger.h"

#define BLOCK_SIZE 256

// t1 = H1 * q1;
__global__ void modalPlateT1Kernel(double *H1, double *q1, double *t1, int nrow, int ncol)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int i;
    double val = 0.0;

    if (idx < nrow) {
	for (i = 0; i < ncol; i++) {
	    val += q1[i] * H1[(i*nrow) + idx];
	}
	t1[idx] = val;
    }
}


// t2 = t1 * q1;
__global__ void modalPlateT2Kernel(double *t1, double *q1, double *t2, int nrow, int ncol)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int i;
    double val = 0.0;

    if (idx < nrow) {
	for (i = 0; i < ncol; i++) {
	    val += q1[i] * t1[(i*nrow) + idx];
	}
	t2[idx] = val;
    }
}

// compute G and q
__global__ void modalPlateQKernel(double *t1, double *t2, double *C, double *C1, double *C2,
			double *q, double *q1, double *q2, int A,
			int DIM)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int i;
    double G = 0.0;

    if (idx < DIM) {
	// G = t1.'*t2;
	for (i = 0; i < A; i++) {
	    G += t2[i] * t1[(idx*A) + i];
	}

	// G = G / C;
	G = G / C[idx];

	// q = -C1*q1 - C2*q2 - G + f_time;
	q[idx] = -C1[idx]*q1[idx] - C2[idx]*q2[idx] - G;
    }
}

GPUModalPlate::GPUModalPlate(int A, int DIM, double *H1, double *C, double *C1,
			     double *C2, double **u, double **u1, double **u2)
{
    ok = false;

    d_u = NULL;
    d_u1 = NULL;
    d_u2 = NULL;

    d_H1 = NULL;
    d_C = NULL;
    d_C1 = NULL;
    d_C2 = NULL;
    d_G = NULL;
    d_t1 = NULL;
    d_t2 = NULL;

    this->A = A;
    this->DIM = DIM;

    // allocate GPU buffers
    cudaMalloc((void **)&d_u, DIM * sizeof(double));
    cudaMalloc((void **)&d_u1, DIM * sizeof(double));
    cudaMalloc((void **)&d_u2, DIM * sizeof(double));

    cudaMalloc((void **)&d_H1, DIM * DIM * A * sizeof(double));
    cudaMalloc((void **)&d_C, DIM * sizeof(double));
    cudaMalloc((void **)&d_C1, DIM * sizeof(double));
    cudaMalloc((void **)&d_C2, DIM * sizeof(double));
    cudaMalloc((void **)&d_G, DIM * sizeof(double));
    cudaMalloc((void **)&d_t1, A * DIM * sizeof(double));
    cudaMalloc((void **)&d_t2, A * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	return;
    }

    // initialise state
    cudaMemset(d_u, 0, DIM * sizeof(double));
    cudaMemset(d_u1, 0, DIM * sizeof(double));
    cudaMemset(d_u2, 0, DIM * sizeof(double));

    cudaMemset(d_G, 0, DIM * sizeof(double));
    cudaMemset(d_t1, 0, A * DIM * sizeof(double));
    cudaMemset(d_t2, 0, A * sizeof(double));
    if (checkCUDAError("cudaMemset")) {
	return;
    }

    // copy constant arrays to GPU
    cudaMemcpy(d_H1, H1, DIM * DIM * A * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, C1, DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2, C2, DIM * sizeof(double), cudaMemcpyHostToDevice);
    if (checkCUDAError("cudaMemcpy")) {
	return;
    }

    // determine grid and block sizes
    block1 = BLOCK_SIZE;
    grid1 = (DIM*DIM*A) / BLOCK_SIZE;
    if (((DIM*DIM*A)%(BLOCK_SIZE)) != 0) grid1++;

    if (A > BLOCK_SIZE) {
	block2 = BLOCK_SIZE;
	grid2 = A / BLOCK_SIZE;
	if ((A%BLOCK_SIZE) != 0) grid2++;
    }
    else {
	block2 = A;
	grid2 = 1;
    }

    if (DIM > BLOCK_SIZE) {
	block3 = BLOCK_SIZE;
	grid3 = DIM / BLOCK_SIZE;
	if ((DIM%BLOCK_SIZE) != 0) grid3++;
    }
    else {
	block3 = DIM;
	grid3 = 1;
    }


    *u = d_u;
    *u1 = d_u1;
    *u2 = d_u2;
    ok = true;
}

GPUModalPlate::~GPUModalPlate()
{
    if (d_u)  cudaFree(d_u);
    if (d_u1) cudaFree(d_u1);
    if (d_u2) cudaFree(d_u2);
    if (d_H1) cudaFree(d_H1);
    if (d_C)  cudaFree(d_C);
    if (d_C1) cudaFree(d_C1);
    if (d_C2) cudaFree(d_C2);
    if (d_G)  cudaFree(d_G);
    if (d_t1) cudaFree(d_t1);
    if (d_t2) cudaFree(d_t2);
}

void GPUModalPlate::runTimestep(int n, double *u, double *u1, double *u2)
{
    dim3 dg1(grid1);
    dim3 db1(block1);
    modalPlateT1Kernel<<<dg1, db1>>>(d_H1, u1, d_t1, A*DIM, DIM);

    dim3 dg2(grid2);
    dim3 db2(block2);
    modalPlateT2Kernel<<<dg2, db2>>>(d_t1, u1, d_t2, A, DIM);

    dim3 dg3(grid3);
    dim3 db3(block3);
    modalPlateQKernel<<<dg3, db3>>>(d_t1, d_t2, d_C, d_C1, d_C2, u, u1,
				    u2, A, DIM);
}

