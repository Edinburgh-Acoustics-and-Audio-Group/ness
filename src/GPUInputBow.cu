/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "GPUInputBow.h"
#include "GPUUtil.h"

#include <cstdlib>
using namespace std;

// pre-computed sqrt(2.0 * exp(1.0))
#define SQRT_2_EXP_1 2.33164398159712416003

// n should already be adjusted to the start of the bow
__global__ void gpuInputBowKernel(double *d_s, double *d_s2, double k, int n, double *velocity,
				  double *force, double jb, int itnum, double sigma, int idx)
{
    int i;
    double g = (1.0 / (2.0 * k)) * (d_s[idx] - d_s2[idx]);
    double alpha = 0.0;
    double vfac = 0.0;

    g -= velocity[n];
    alpha = jb * force[n];
    vfac = velocity[n];

    double v = 0.0;
    for (i = 0; i < itnum; i++) {
	v = v - (v - g + SQRT_2_EXP_1 * alpha * sigma * v *
		 exp(-(sigma*sigma)*(v*v))) / (1.0 + SQRT_2_EXP_1 * alpha * sigma *
					       (1.0 - 2.0*(sigma*sigma)*(v*v)) *
					       exp(-(sigma*sigma)*(v*v)));
    }

    d_s[idx] = d_s2[idx] + 2.0 * k * (v + vfac);    
}

GPUInputBow::GPUInputBow(int len, double *velocity, double *force, double k, double jb,
			 int itnum, double sigma, int index)
{
    this->k = k;
    this->jb = jb;
    this->itnum = itnum;
    this->sigma = sigma;
    this->index = index;

    cudaMalloc((void **)&this->velocity, len * sizeof(double));
    cudaMalloc((void **)&this->force, len * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    cudaMemcpy(this->velocity, velocity, len * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(this->force, force, len * sizeof(double), cudaMemcpyHostToDevice);
}

GPUInputBow::~GPUInputBow()
{
    cudaFree(this->velocity);
    cudaFree(this->force);
}

void GPUInputBow::runTimestep(int n, double *s, double *s2)
{
    dim3 dimGrid(1);
    dim3 dimBlock(1);
    gpuInputBowKernel<<<dimGrid, dimBlock>>>(s, s2, k, n, velocity, force, jb, itnum, sigma, index);
}

