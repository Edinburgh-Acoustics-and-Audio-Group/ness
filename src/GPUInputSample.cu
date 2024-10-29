/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "GPUInputSample.h"
#include "GPUUtil.h"

#include <cstdlib>
using namespace std;

__global__ void gpuInputSampleKernel(double *d_data, double *d_s, int index, int n, double multiplier)
{
    d_s[index] += multiplier * d_data[n];
}

__global__ void gpuInputSampleKernelLinear(double *d_data, double *d_s, int index, int n, double multiplier,
					   double a0, double a1)
{
    d_s[index] += multiplier * a0 * d_data[n];
    d_s[index+1] += multiplier * a1 * d_data[n];
}

__global__ void gpuInputSampleKernelBilinear(double *d_data, double *d_s, int index, int n, double multiplier,
					     double a0, double a1, double a2, double a3, int nx)
{
    d_s[index] += multiplier * a0 * d_data[n];
    d_s[index+1] += multiplier * a1 * d_data[n];
    d_s[index+nx] += multiplier * a2 * d_data[n];
    d_s[index+nx+1] += multiplier * a3 * d_data[n];
}

// FIXME: could speed this up by running it on 8 threads
__global__ void gpuInputSampleKernelTrilinear(double *d_data, double *d_s, int index, int n, double multiplier,
					      double a0, double a1, double a2, double a3, double a4, double a5,
					      double a6, double a7, int nx, int nxny)
{
    d_s[index] += multiplier * a0 * d_data[n];
    d_s[index+1] += multiplier * a1 * d_data[n];
    d_s[index+nx] += multiplier * a2 * d_data[n];
    d_s[index+nx+1] += multiplier * a3 * d_data[n];
    d_s[index+nxny] += multiplier * a4 * d_data[n];
    d_s[index+nxny+1] += multiplier * a5 * d_data[n];
    d_s[index+nxny+nx] += multiplier * a6 * d_data[n];
    d_s[index+nxny+nx+1] += multiplier * a7 * d_data[n];
}

// interp points into the main InputSample object, so don't delete it!
GPUInputSample::GPUInputSample(double *data, int len, double multiplier, int index, InterpolationInfo *interp)
{
    this->multiplier = multiplier;
    this->index = index;
    this->interp = interp;

    cudaMalloc((void **)&d_data, len * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    cudaMemcpy(d_data, data, len * sizeof(double), cudaMemcpyHostToDevice);
}

GPUInputSample::~GPUInputSample()
{
    cudaFree(d_data);
}

// n should be relative to sample here!
void GPUInputSample::runTimestep(int n, double *s)
{
    dim3 dimGrid(1);
    dim3 dimBlock(1);
    switch (interp->type) {
    case INTERPOLATION_NONE:
	gpuInputSampleKernel<<<dimGrid, dimBlock>>>(d_data, s, index, n, multiplier);
	break;
    case INTERPOLATION_LINEAR:
	gpuInputSampleKernelLinear<<<dimGrid, dimBlock>>>(d_data, s, index, n, multiplier,
							  interp->alpha[0], interp->alpha[1]);
	break;
    case INTERPOLATION_BILINEAR:
	gpuInputSampleKernelBilinear<<<dimGrid, dimBlock>>>(d_data, s, index, n, multiplier,
							    interp->alpha[0], interp->alpha[1],
							    interp->alpha[2], interp->alpha[3],
							    interp->nx);
	break;
    case INTERPOLATION_TRILINEAR:
	gpuInputSampleKernelTrilinear<<<dimGrid, dimBlock>>>(d_data, s, index, n, multiplier,
							     interp->alpha[0], interp->alpha[1],
							     interp->alpha[2], interp->alpha[3],
							     interp->alpha[4], interp->alpha[5],
							     interp->alpha[6], interp->alpha[7],
							     interp->nx, interp->nxny);
	break;
    }
}

