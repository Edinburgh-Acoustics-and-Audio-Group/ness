/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyComponent2D.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

DummyComponent2D::DummyComponent2D(string name, int nx, int ny)
    : Component2D(name)
{
    allocateState(nx, ny);
    alpha = 1.0;
    bowFactor = 1.0;
    onGPU = false;
}

DummyComponent2D::~DummyComponent2D()
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

void DummyComponent2D::runTimestep(int n)
{
}

bool DummyComponent2D::isOnGPU()
{
    return onGPU;
}

bool DummyComponent2D::moveToGPU()
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
