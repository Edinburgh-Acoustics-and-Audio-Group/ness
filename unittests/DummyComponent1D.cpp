/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyComponent1D.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

DummyComponent1D::DummyComponent1D(string name, int n)
    : Component1D(name)
{
    allocateState(n);
    alpha = 1.0;
    bowFactor = 1.0;
    onGPU = false;
}

DummyComponent1D::~DummyComponent1D()
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

void DummyComponent1D::runTimestep(int n)
{
}

bool DummyComponent1D::isOnGPU()
{
    return onGPU;
}

bool DummyComponent1D::moveToGPU()
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
