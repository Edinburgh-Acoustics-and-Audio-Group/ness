/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "GlobalSettings.h"
#include "GPUAirboxIndexed.h"
#include "GPUUtil.h"

static int BLOCK_W = 8;
static int BLOCK_H = 8;
static int BLOCK_D = 8;


__global__ void gpuAirboxIndexedKernel(double *Psi0, double *Psi1, double *Psi, int nx, int ny, int nz,
				       unsigned char *index, double *coeffs, int basez)
{
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    // +1 for skipping the halo
    int z = threadIdx.z + (blockIdx.z * blockDim.z) + basez + 1;

    if ((x < nx) && (y < ny) && (z <= nz)) {
	int nxny = nx * ny;

	int i = ((z * nxny) + y + (x * ny));

	// no halo on index
	double *c = &coeffs[index[i - nxny] << 4];
	Psi[i] = c[8] * ((Psi1[i]*c[0] + Psi1[i+ny]*c[1] + Psi1[i-ny]*c[2] +
			  Psi1[i+1]*c[3] + Psi1[i-1]*c[4] + Psi1[i+nxny]*c[5] +
			  Psi1[i-nxny]*c[6]) -
			 (Psi0[i]*c[7] + Psi0[i+ny]*c[9] + Psi0[i-ny]*c[10] +
			  Psi0[i+1]*c[11] + Psi0[i-1]*c[12] + Psi0[i+nxny]*c[13] +
			  Psi0[i-nxny]*c[14]));
    }
}

GPUAirboxIndexed::GPUAirboxIndexed(int nx, int ny, int nz, double **u, double **u1,
				   double **u2, unsigned char *h_index, double *h_coeffs)
{
    ok = false;
    d_u = NULL;
    d_u1 = NULL;
    d_u2 = NULL;
    index = NULL;
    coeffs = NULL;

    this->nx = nx;
    this->ny = ny;
    this->nz = nz;

    int ss = nx*ny*nz;
    int sshalo = nx*ny*(nz+2);

    // allocate state arrays
    cudaMalloc((void **)&d_u, sshalo * sizeof(double));
    cudaMalloc((void **)&d_u1, sshalo * sizeof(double));
    cudaMalloc((void **)&d_u2, sshalo * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	return;
    }

    // initialise state arrays
    cudaMemset(d_u, 0, sshalo * sizeof(double));
    cudaMemset(d_u1, 0, sshalo * sizeof(double));
    cudaMemset(d_u2, 0, sshalo * sizeof(double));

    // pass state arrays back to caller
    *u = d_u;
    *u1 = d_u1;
    *u2 = d_u2;

    // allocate index and co-efficient arrays
    cudaMalloc((void **)&index, ss);
    cudaMalloc((void **)&coeffs, 16*256*sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	return;
    }

    // initialise index and co-efficient arrays
    cudaMemcpy(index, h_index, ss, cudaMemcpyHostToDevice);
    cudaMemcpy(coeffs, h_coeffs, 16*256*sizeof(double), cudaMemcpyHostToDevice);

    // calculate grid dimensions
    GlobalSettings *gs = GlobalSettings::getInstance();
    BLOCK_W = gs->getCuda3dBlockW();
    BLOCK_H = gs->getCuda3dBlockH();
    BLOCK_D = gs->getCuda3dBlockD();

    gridW = nx / BLOCK_W;
    if ((nx % BLOCK_W) != 0) gridW++;
    gridH = ny / BLOCK_H;
    if ((ny % BLOCK_H) != 0) gridH++;
    gridD = nz / BLOCK_D;
    if ((nz % BLOCK_D) != 0) gridD++;

    ok = true;
}

GPUAirboxIndexed::~GPUAirboxIndexed()
{
    if (d_u) cudaFree(d_u);
    if (d_u1) cudaFree(d_u1);
    if (d_u2) cudaFree(d_u2);
    if (index) cudaFree(index);
    if (coeffs) cudaFree(coeffs);
}

void GPUAirboxIndexed::runTimestep(int n, double *u, double *u1, double *u2)
{
    dim3 dimGrid(gridW, gridH, gridD);
    dim3 dimBlock(BLOCK_W, BLOCK_H, BLOCK_D);
    gpuAirboxIndexedKernel<<<dimGrid, dimBlock>>>(u2, u1, u, nx, ny, nz, index, coeffs, 0);
}

void GPUAirboxIndexed::runPartialUpdate(double *u, double *u1, double *u2, int start, int len)
{
    dim3 dimGrid(gridW, gridH, 1);
    dim3 dimBlock(BLOCK_W, BLOCK_H, len);
    gpuAirboxIndexedKernel<<<dimGrid, dimBlock>>>(u2, u1, u, nx, ny, nz, index, coeffs, start);
}

