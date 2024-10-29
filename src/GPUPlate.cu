/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "GPUPlate.h"
#include "Logger.h"
#include "GPUUtil.h"
#include "GlobalSettings.h"

extern "C" {
#include "matrix_unroll.h"
};

#include <cstdlib>
#include <cmath>
using namespace std;

static int BLOCK_W = 16;
static int BLOCK_H = 16;

/*
 * Indices of the stencil elements within each co-efficient set
 */
#define STENCIL_FARLEFT     0
#define STENCIL_UP_LEFT     1
#define STENCIL_LEFT        2
#define STENCIL_DOWN_LEFT   3
#define STENCIL_TOP         4
#define STENCIL_UP          5
#define STENCIL_MIDDLE      6
#define STENCIL_DOWN        7
#define STENCIL_BOTTOM      8
#define STENCIL_UP_RIGHT    9
#define STENCIL_RIGHT      10
#define STENCIL_DOWN_RIGHT 11
#define STENCIL_FARRIGHT   12

__global__ void gpuPlateKernel(double *d_u, double *d_u1, double *d_u2, int nx, int ny,
			       unsigned char *d_indexb, unsigned char *d_indexc, double *coeffsb,
			       double *coeffsc)
{
    /* work out which element we're working on */
    int x = threadIdx.y + (blockIdx.y * blockDim.y);
    int y = threadIdx.x + (blockIdx.x * blockDim.x);

    int idx = (x * ny) + y;
    int ny2 = ny + ny;
    double result;

    if ((x < nx) && (y < ny)) {
	int bcbase = d_indexb[idx] * 13; /* look up co-efficient indexes */
	int ccbase = d_indexc[idx] * 13;
	int uidx = idx + ny2; /* take the "halo" into account */
	result =
	    coeffsb[bcbase + STENCIL_FARLEFT] * d_u1[uidx - ny2] +
	    coeffsb[bcbase + STENCIL_UP_LEFT] * d_u1[uidx - ny - 1] +
	    coeffsb[bcbase + STENCIL_LEFT] * d_u1[uidx - ny] +
	    coeffsb[bcbase + STENCIL_DOWN_LEFT] * d_u1[uidx - ny + 1] +
	    coeffsb[bcbase + STENCIL_TOP] * d_u1[uidx - 2] +
	    coeffsb[bcbase + STENCIL_UP] * d_u1[uidx - 1] +
	    coeffsb[bcbase + STENCIL_MIDDLE] * d_u1[uidx] +
	    coeffsb[bcbase + STENCIL_DOWN] * d_u1[uidx + 1] +
	    coeffsb[bcbase + STENCIL_BOTTOM] * d_u1[uidx + 2] +
	    coeffsb[bcbase + STENCIL_UP_RIGHT] * d_u1[uidx + ny - 1] +
	    coeffsb[bcbase + STENCIL_RIGHT] * d_u1[uidx + ny] +
	    coeffsb[bcbase + STENCIL_DOWN_RIGHT] * d_u1[uidx + ny + 1] +
	    coeffsb[bcbase + STENCIL_FARRIGHT] * d_u1[uidx + ny2] +
	    coeffsc[ccbase + STENCIL_LEFT] * d_u2[uidx - ny] +
	    coeffsc[ccbase + STENCIL_UP] * d_u2[uidx - 1] +
	    coeffsc[ccbase + STENCIL_MIDDLE] * d_u2[uidx] +
	    coeffsc[ccbase + STENCIL_DOWN] * d_u2[uidx + 1] +
	    coeffsc[ccbase + STENCIL_RIGHT] * d_u2[uidx + ny];
	d_u[uidx] = result;
    }    
}

GPUPlate::GPUPlate(int nx, int ny, CSRmatrix *B, CSRmatrix *C, double **u, double **u1, double **u2)
{
    int ncoeffsb, ncoeffsc;

    double *h_coeffsb, *h_coeffsc;
    unsigned char *h_indexb, *h_indexc;

    indexb = NULL;
    indexc = NULL;
    coeffsb = NULL;
    coeffsc = NULL;
    d_u = NULL;
    d_u1 = NULL;
    d_u2 = NULL;

    ok = false;

    this->nx = nx;
    this->ny = ny;

    // unroll both matrices
    h_indexb = new unsigned char[nx * ny];
    h_indexc = new unsigned char[nx * ny];

    if (!unrollMatrixWithIndex(B, nx-1, ny-1, &ncoeffsb, &h_coeffsb, h_indexb)) {
	delete[] h_indexb;
	delete[] h_indexc;
	return;
    }

    if (!unrollMatrixWithIndex(C, nx-1, ny-1, &ncoeffsc, &h_coeffsc, h_indexc)) {
	delete[] h_indexb;
	delete[] h_indexc;
	return;
    }

    logMessage(1, "B coefficient sets: %d, C coefficient sets: %d", ncoeffsb, ncoeffsc);

    // allocate space on GPU for the unrolled matrices
    cudaMalloc((void **)&indexb, nx * ny);
    cudaMalloc((void **)&indexc, nx * ny);
    cudaMalloc((void **)&coeffsb, 13 * ncoeffsb * sizeof(double));
    cudaMalloc((void **)&coeffsc, 13 * ncoeffsc * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	return;
    }

    // copy unrolled matrices to GPU
    cudaMemcpy(indexb, h_indexb, nx * ny, cudaMemcpyHostToDevice);
    cudaMemcpy(indexc, h_indexc, nx * ny, cudaMemcpyHostToDevice);
    cudaMemcpy(coeffsb, h_coeffsb, 13 * ncoeffsb * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(coeffsc, h_coeffsc, 13 * ncoeffsc * sizeof(double), cudaMemcpyHostToDevice);

    // allocate and initialise state arrays - remember we need a halo!
    int d_ss = (nx+4) * ny;
    cudaMalloc((void **)&d_u, d_ss * sizeof(double));
    cudaMalloc((void **)&d_u1, d_ss * sizeof(double));
    cudaMalloc((void **)&d_u2, d_ss * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	return;
    }
    cudaMemset(d_u, 0, d_ss * sizeof(double));
    cudaMemset(d_u1, 0, d_ss * sizeof(double));
    cudaMemset(d_u2, 0, d_ss * sizeof(double));

    // return state array pointers to caller so it can stash them in Plate object
    *u = d_u;
    *u1 = d_u1;
    *u2 = d_u2;

    delete[] h_indexb;
    delete[] h_indexc;
    free(h_coeffsb);
    free(h_coeffsc);

    // compute grid width and height
    GlobalSettings *gs = GlobalSettings::getInstance();
    BLOCK_W = gs->getCuda2dBlockW();
    BLOCK_H = gs->getCuda2dBlockH();
    gridW = ny / BLOCK_W;
    if ((ny % BLOCK_W) != 0) gridW++;
    gridH = nx / BLOCK_H;
    if ((nx % BLOCK_H) != 0) gridH++;

    ok = true;
}

GPUPlate::~GPUPlate()
{
    if (indexb) cudaFree(indexb);
    if (indexc) cudaFree(indexc);
    if (coeffsb) cudaFree(coeffsb);
    if (coeffsc) cudaFree(coeffsc);
    if (d_u) cudaFree(d_u);
    if (d_u1) cudaFree(d_u1);
    if (d_u2) cudaFree(d_u2);
}

void GPUPlate::runTimestep(int n, double *u, double *u1, double *u2)
{
    dim3 dimGrid(gridW, gridH);
    dim3 dimBlock(BLOCK_W, BLOCK_H);
    gpuPlateKernel<<<dimGrid, dimBlock>>>(u, u1, u2, nx, ny, indexb, indexc, coeffsb, coeffsc);
}
