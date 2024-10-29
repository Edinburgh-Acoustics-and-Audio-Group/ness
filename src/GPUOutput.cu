/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "GPUOutput.h"
#include "GlobalSettings.h"
#include "GPUUtil.h"

#include <cstdlib>
using namespace std;

__global__ void gpuOutputKernelNormal(double *d_data, double *d_u, int idx, int n)
{
    d_data[n] = d_u[idx];
}

__global__ void gpuOutputKernelNormalLinear(double *d_data, double *d_u, int idx, int n, double a0, double a1)
{
    d_data[n] = a0*d_u[idx] + a1*d_u[idx+1];
}

__global__ void gpuOutputKernelNormalBilinear(double *d_data, double *d_u, int idx, int n, double a0, double a1,
					      double a2, double a3, int nx)
{
    d_data[n] = a0*d_u[idx] + a1*d_u[idx+1] + a2*d_u[idx+nx] + a3*d_u[idx+nx+1];
}

__global__ void gpuOutputKernelNormalTrilinear(double *d_data, double *d_u, int idx, int n, double a0, double a1,
					       double a2, double a3, double a4, double a5, double a6, double a7,
					       int nx, int nxny)
{
    d_data[n] = a0*d_u[idx] + a1*d_u[idx+1] + a2*d_u[idx+nx] + a3*d_u[idx+nx+1] +
	a4*d_u[idx+nxny] + a5*d_u[idx+nxny+1] + a6*d_u[idx+nxny+nx] + a7*d_u[idx+nxny+nx+1];
}


__global__ void gpuOutputKernelDifference(double *d_data, double *d_u, double *d_u1, int idx, int n, double SR)
{
    d_data[n] = SR * (d_u[idx] - d_u1[idx]);
}

__global__ void gpuOutputKernelDifferenceLinear(double *d_data, double *d_u, double *d_u1, int idx, int n,
						double SR, double a0, double a1)
{
    double uval = a0*d_u[idx] + a1*d_u[idx+1];
    double u1val = a0*d_u1[idx] + a1*d_u1[idx+1];
    d_data[n] = SR * (uval - u1val);
}

__global__ void gpuOutputKernelDifferenceBilinear(double *d_data, double *d_u, double *d_u1, int idx, int n,
						  double SR, double a0, double a1, double a2, double a3,
						  int nx)
{
    double uval = a0*d_u[idx] + a1*d_u[idx+1] + a2*d_u[idx+nx] + a3*d_u[idx+nx+1];
    double u1val = a0*d_u1[idx] + a1*d_u1[idx+1] + a2*d_u1[idx+nx] + a3*d_u1[idx+nx+1];
    d_data[n] = SR * (uval - u1val);
}

__global__ void gpuOutputKernelDifferenceTrilinear(double *d_data, double *d_u, double *d_u1, int idx, int n,
						   double SR, double a0, double a1, double a2, double a3,
						   double a4, double a5, double a6, double a7, int nx, int nxny)
{
    double uval = a0*d_u[idx] + a1*d_u[idx+1] + a2*d_u[idx+nx] + a3*d_u[idx+nx+1] +
	a4*d_u[idx+nxny] + a5*d_u[idx+nxny+1] + a6*d_u[idx+nxny+nx] + a7*d_u[idx+nxny+nx+1];
    double u1val = a0*d_u1[idx] + a1*d_u1[idx+1] + a2*d_u1[idx+nx] + a3*d_u1[idx+nx+1] +
	a4*d_u1[idx+nxny] + a5*d_u1[idx+nxny+1] + a6*d_u1[idx+nxny+nx] + a7*d_u1[idx+nxny+nx+1];
    d_data[n] = SR * (uval - u1val);
}



GPUOutput::GPUOutput(int type, Component *comp, int idx, InterpolationInfo *interp)
{
    // combine the interpolation and output types so we can do a single switch on both
    this->type = (type | (interp->type << 2));
    component = comp;
    this->idx = idx;
    this->interp = interp;

    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    cudaMalloc((void **)&d_data, NF * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    cudaMemset(d_data, 0, NF * sizeof(double));
}

GPUOutput::GPUOutput(Component *comp)
{
    this->component = comp;
}

GPUOutput::~GPUOutput()
{
    cudaFree(d_data);
}

void GPUOutput::runTimestep(int n)
{
    dim3 gridDim(1);
    dim3 blockDim(1);

    switch (type) {
    case (GPUOUTPUTTYPE_NORMAL | (INTERPOLATION_NONE << 2)):
	gpuOutputKernelNormal<<<gridDim, blockDim>>>(d_data, component->getU(), idx, n);
	break;
    case (GPUOUTPUTTYPE_NORMAL | (INTERPOLATION_LINEAR << 2)):
	gpuOutputKernelNormalLinear<<<gridDim, blockDim>>>(d_data, component->getU(), idx, n, interp->alpha[0],
							   interp->alpha[1]);
	break;
    case (GPUOUTPUTTYPE_NORMAL | (INTERPOLATION_BILINEAR << 2)):
	gpuOutputKernelNormalBilinear<<<gridDim, blockDim>>>(d_data, component->getU(), idx, n, interp->alpha[0],
							     interp->alpha[1], interp->alpha[2], interp->alpha[3],
							     interp->nx);
	break;
    case (GPUOUTPUTTYPE_NORMAL | (INTERPOLATION_TRILINEAR << 2)):
	gpuOutputKernelNormalTrilinear<<<gridDim, blockDim>>>(d_data, component->getU(), idx, n, interp->alpha[0],
							      interp->alpha[1], interp->alpha[2], interp->alpha[3],
							      interp->alpha[4], interp->alpha[5], interp->alpha[6],
							      interp->alpha[7], interp->nx, interp->nxny);
	break;

    case (GPUOUTPUTTYPE_DIFFERENCE | (INTERPOLATION_NONE << 2)):
	gpuOutputKernelDifference<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU1(),
							 idx, n, GlobalSettings::getInstance()->getSampleRate());
	break;
    case (GPUOUTPUTTYPE_DIFFERENCE | (INTERPOLATION_LINEAR << 2)):
	gpuOutputKernelDifferenceLinear<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU1(),
							       idx, n,
							       GlobalSettings::getInstance()->getSampleRate(),
							       interp->alpha[0], interp->alpha[1]);
	break;
    case (GPUOUTPUTTYPE_DIFFERENCE | (INTERPOLATION_BILINEAR << 2)):
	gpuOutputKernelDifferenceBilinear<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU1(),
								 idx, n,
								 GlobalSettings::getInstance()->getSampleRate(),
								 interp->alpha[0], interp->alpha[1],
								 interp->alpha[2], interp->alpha[3], interp->nx);
	break;
    case (GPUOUTPUTTYPE_DIFFERENCE | (INTERPOLATION_TRILINEAR << 2)):
	gpuOutputKernelDifferenceTrilinear<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU1(),
								  idx, n,
								  GlobalSettings::getInstance()->getSampleRate(),
								  interp->alpha[0], interp->alpha[1],
								  interp->alpha[2], interp->alpha[3], 
								  interp->alpha[4], interp->alpha[5],
								  interp->alpha[6], interp->alpha[7], interp->nx,
								  interp->nxny);
	break;

    case (GPUOUTPUTTYPE_PRESSURE | (INTERPOLATION_NONE << 2)):
	gpuOutputKernelDifference<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU2(),
							 idx, n, GlobalSettings::getInstance()->getSampleRate());
	break;
    case (GPUOUTPUTTYPE_PRESSURE | (INTERPOLATION_LINEAR << 2)):
	gpuOutputKernelDifferenceLinear<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU2(),
							       idx, n,
							       GlobalSettings::getInstance()->getSampleRate(),
							       interp->alpha[0], interp->alpha[1]);
	break;
    case (GPUOUTPUTTYPE_PRESSURE | (INTERPOLATION_BILINEAR << 2)):
	gpuOutputKernelDifferenceBilinear<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU2(),
								 idx, n,
								 GlobalSettings::getInstance()->getSampleRate(),
								 interp->alpha[0], interp->alpha[1],
								 interp->alpha[2], interp->alpha[3], interp->nx);
	break;
    case (GPUOUTPUTTYPE_PRESSURE | (INTERPOLATION_TRILINEAR << 2)):
	gpuOutputKernelDifferenceTrilinear<<<gridDim, blockDim>>>(d_data, component->getU(), component->getU2(),
								  idx, n,
								  GlobalSettings::getInstance()->getSampleRate(),
								  interp->alpha[0], interp->alpha[1],
								  interp->alpha[2], interp->alpha[3], 
								  interp->alpha[4], interp->alpha[5],
								  interp->alpha[6], interp->alpha[7], interp->nx,
								  interp->nxny);
	break;
    }
}

void GPUOutput::getData(double *buf)
{
    cudaMemcpy(buf, d_data, GlobalSettings::getInstance()->getNumTimesteps() * sizeof(double),
	       cudaMemcpyDeviceToHost);
}

