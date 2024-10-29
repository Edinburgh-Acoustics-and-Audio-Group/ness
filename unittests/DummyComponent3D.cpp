/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyComponent3D.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

DummyComponent3D::DummyComponent3D(string name, int nx, int ny, int nz)
    : Component3D(name)
{
    allocateState(nx, ny, nz);
    onGPU = false;
    alpha = 1.0;
    bowFactor = 1.0;
}

DummyComponent3D::~DummyComponent3D()
{
#ifdef USE_GPU
    if (onGPU) {
	cudaFreeNess(u);
	cudaFreeNess(u1);
	cudaFreeNess(u2);
	u = NULL;
	u1 = NULL;
	u2 = NULL;
    }
#endif
}

void DummyComponent3D::runTimestep(int n)
{
}

bool DummyComponent3D::isOnGPU()
{
    return onGPU;
}

bool DummyComponent3D::moveToGPU()
{
#ifdef USE_GPU
    delete[] u;
    delete[] u1;
    delete[] u2;

    u = (double *)cudaMallocNess(ss * sizeof(double));
    u1 = (double *)cudaMallocNess(ss * sizeof(double));
    u2 = (double *)cudaMallocNess(ss * sizeof(double));

    cudaMemsetNess(u, 0, ss * sizeof(double));
    cudaMemsetNess(u1, 0, ss * sizeof(double));
    cudaMemsetNess(u2, 0, ss * sizeof(double));

    onGPU = true;
#endif
    return onGPU;
}
