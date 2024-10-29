/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "GPUInputModalStrike.h"
#include "GPUUtil.h"
#include "GlobalSettings.h"
#include "MathUtil.h"

#define BLOCK_SIZE 256

__global__ void gpuModalStrikeKernel(double *d_p, double *s, double val, int DIM)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (idx < DIM) {
	s[idx] += val * d_p[idx];
    }
}

GPUInputModalStrike::GPUInputModalStrike(double *P, int startTime, int duration,
					 double amplitude, int DIM, double T0,
					 double Twid)
{
    this->amplitude = amplitude;
    this->DIM = DIM;
    this->startTime = startTime;
    this->duration = duration;
    this->T0 = T0;
    this->Twid = Twid;

    cudaMalloc((void **)&d_p, DIM * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    cudaMemcpy(d_p, P, DIM * sizeof(double), cudaMemcpyHostToDevice);
    if (checkCUDAError("cudaMemcpy")) {
	exit(1);
    }
}

GPUInputModalStrike::~GPUInputModalStrike()
{
    cudaFree(d_p);
}

void GPUInputModalStrike::runTimestep(int n, double *s)
{
    if ((n >= startTime) && (n < (startTime+duration))) {
	int dg = (DIM / BLOCK_SIZE);
	int db = BLOCK_SIZE;
	if ((DIM % db) != 0) dg++;
	dim3 dimgrid(dg);
	dim3 dimblock(db);

	double dn = (double)n;
	double fs = GlobalSettings::getInstance()->getSampleRate();
	double val = 0.5 * amplitude * (1.0 + cos(M_PI * (dn/fs/Twid - T0/Twid)));
	gpuModalStrikeKernel<<<dimgrid, dimblock>>>(d_p, s, val, DIM);
    }
}

