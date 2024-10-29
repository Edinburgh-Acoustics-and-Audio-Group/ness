/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "GPUOutputModal.h"
#include "GlobalSettings.h"
#include "GPUUtil.h"
#include "Logger.h"

// kernel
__global__ void gpuModalOutputKernel(double *d_data, double *d_u, double *d_rp,
				     int n, double h, int DIM)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < DIM; i++) {
	sum += d_rp[i] * d_u[i];
    }
    d_data[n] = sum / h;
}

GPUOutputModal::GPUOutputModal(ModalPlate *comp, double *rp)
    : GPUOutput(comp)
{
    DIM = comp->getStateSize();
    h = comp->getH();

    // allocate d_rp and d_data
    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    cudaMalloc((void **)&d_data, NF * sizeof(double));
    cudaMalloc((void **)&d_rp, DIM * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    cudaMemset(d_data, 0, NF * sizeof(double));

    // copy rp to device
    cudaMemcpy(d_rp, rp, DIM * sizeof(double), cudaMemcpyHostToDevice);
}

GPUOutputModal::~GPUOutputModal()
{
    // free d_rp
    cudaFree(d_rp);
}

void GPUOutputModal::runTimestep(int n)
{
    // invoke kernel
    dim3 gridDim(1);
    dim3 blockDim(1);
    gpuModalOutputKernel<<<gridDim, blockDim>>>(d_data, component->getU(), d_rp,
						n, h, DIM);
}

