/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "GPUEmbedding.h"
#include "Logger.h"
#include "GlobalSettings.h"

#include <cuda_runtime.h>

#include <cstdlib>
using namespace std;

static int BLOCK_SIZE = 256;


// interpolation from plate back to airbox
__global__ void gpuEmbeddingKernelReverse(int n, double *Diffstar_n, double *Diff_n, double *Diff_tmp,
					  double *Diff_np, double *Sum_np, double *psidown, double *psiup,
					  double *true_Psi, double Qdivk, double Gammasq)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < n) {
	Diffstar_n[i] = -(true_Psi[i]*Diff_n[i]) + (Qdivk*Diff_tmp[i]);
	Diff_np[i] = Diff_np[i] - (Gammasq*Diffstar_n[i]) +
	    (Gammasq*true_Psi[i]*Diff_n[i]);

	if (true_Psi[i] > 0.5) {
	    psidown[i] = 0.5 * (Sum_np[i] + Diff_np[i]);
	    psiup[i] = 0.5 * (Sum_np[i] - Diff_np[i]);
	}
    }
}

// interpolation from airbox to plate
__global__ void gpuEmbeddingKernelForward(int n, double *Sum_np, double *psidown, double *psiup, double *Diff_np,
					  double *Diffstar_nm, double *Diff_nm, double *Diff_n, double *Diff_tmp,
					  double lambdasq, double diffnfac)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < n) {
	Sum_np[i] = psidown[i] + psiup[i];
	Diff_np[i] = psidown[i] - psiup[i];
	
	Diff_np[i] += lambdasq * (Diffstar_nm[i] - Diff_nm[i]);
	Diff_tmp[i] = Diff_np[i] + (diffnfac * Diff_n[i]) - Diff_nm[i];
    }
}

GPUEmbedding::GPUEmbedding(CSRmatrix *BfIMat, CSRmatrix *JMat, int Diff_size, int pd, int pu, Airbox *airbox,
			   double k, int plateSS, double *h_true_Psi)
{
    this->Diff_size = Diff_size;
    this->plateSS = plateSS;
    this->pd = pd;
    this->pu = pu;

    // initialise scalars
    Qdivk = airbox->getQ() / k;
    Gammasq = airbox->getGamma() * airbox->getGamma();
    lambdasq = airbox->getLambda() * airbox->getLambda();
    diffnfac = 2.0 * Gammasq;

    // convert matrices to GPU format
    this->JMat = csrToGpuMatrix(JMat);
    this->BfIMat = csrToGpuMatrix(BfIMat);

    // allocate buffers
    cudaMalloc((void **)&Sum_np, Diff_size * sizeof(double));
    cudaMalloc((void **)&Diff_np, Diff_size * sizeof(double));
    cudaMalloc((void **)&Diff_n, Diff_size * sizeof(double));
    cudaMalloc((void **)&Diff_nm, Diff_size * sizeof(double));
    cudaMalloc((void **)&Diffstar_n, Diff_size * sizeof(double));
    cudaMalloc((void **)&Diffstar_nm, Diff_size * sizeof(double));
    cudaMalloc((void **)&Diff_tmp, Diff_size * sizeof(double));
    cudaMalloc((void **)&true_Psi, Diff_size * sizeof(double));
    cudaMalloc((void **)&transferBuffer, plateSS * sizeof(double));
    if (checkCUDAError("cudaMalloc")) {
	exit(1);
    }

    // clear buffers
    cudaMemset(Sum_np, 0, Diff_size * sizeof(double));
    cudaMemset(Diff_np, 0, Diff_size * sizeof(double));
    cudaMemset(Diff_n, 0, Diff_size * sizeof(double));
    cudaMemset(Diff_nm, 0, Diff_size * sizeof(double));
    cudaMemset(Diffstar_nm, 0, Diff_size * sizeof(double));
    cudaMemset(Diffstar_n, 0, Diff_size * sizeof(double));
    cudaMemset(Diff_tmp, 0, Diff_size * sizeof(double));
    cudaMemset(transferBuffer, 0, plateSS * sizeof(double));

    cudaMemcpy(true_Psi, h_true_Psi, Diff_size * sizeof(double), cudaMemcpyHostToDevice);

    // compute grid size for kernels
    GlobalSettings *gs = GlobalSettings::getInstance();
    BLOCK_SIZE = gs->getCuda2dBlockW() * gs->getCuda2dBlockH();
    gridSize = Diff_size / BLOCK_SIZE;
    if ((Diff_size % BLOCK_SIZE) != 0) gridSize++;
}

GPUEmbedding::~GPUEmbedding()
{
    // free matrices
    freeGpuMatrix(JMat);
    freeGpuMatrix(BfIMat);

    // free buffers
    cudaFree(Sum_np);
    cudaFree(Diff_np);
    cudaFree(Diff_n);
    cudaFree(Diff_nm);
    cudaFree(Diffstar_nm);
    cudaFree(Diffstar_n);
    cudaFree(Diff_tmp);
    cudaFree(true_Psi);
    cudaFree(transferBuffer);
}

void GPUEmbedding::runTimestep(int n, Airbox *airbox, double *hostBuffer)
{
    double *psiup = airbox->getU1() + pu * Diff_size;
    double *psidown = airbox->getU1() + pd * Diff_size;
    double *tmp;

    // copy plate transfer buffer data to GPU
    cudaMemcpy(transferBuffer, hostBuffer, plateSS * sizeof(double), cudaMemcpyHostToDevice);

    // reverse interpolation from plate to airbox
    // multiply by JMat
    gpuMatrixMultiply(JMat, transferBuffer, Diff_tmp);

    dim3 gridDim(gridSize);
    dim3 blockDim(BLOCK_SIZE);
    gpuEmbeddingKernelReverse<<<gridDim, blockDim>>>(Diff_size, Diffstar_n, Diff_n, Diff_tmp, Diff_np,
						     Sum_np, psidown, psiup, true_Psi, Qdivk, Gammasq);

    // re-run slice of airbox
    airbox->runPartialUpdate(pd - 1, 4);

    // swap buffers
    tmp = Diff_nm;
    Diff_nm = Diff_n;
    Diff_n = Diff_np;
    Diff_np = tmp;

    tmp = Diffstar_nm;
    Diffstar_nm = Diffstar_n;
    Diffstar_n = tmp;

    // forward interpolation from airbox to plate
    psidown = airbox->getU() + pd * Diff_size;
    psiup = airbox->getU() + pu * Diff_size;
    
    gpuEmbeddingKernelForward<<<gridDim, blockDim>>>(Diff_size, Sum_np, psidown, psiup, Diff_np,
						     Diffstar_nm, Diff_nm, Diff_n, Diff_tmp,
						     lambdasq, diffnfac);
    // multiply by BfIMat
    gpuMatrixMultiply(BfIMat, Diff_tmp, transferBuffer);

    // copy data back to host
    cudaMemcpy(hostBuffer, transferBuffer, plateSS * sizeof(double), cudaMemcpyDeviceToHost);
}
