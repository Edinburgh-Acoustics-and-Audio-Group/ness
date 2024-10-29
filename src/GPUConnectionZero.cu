/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "GPUConnectionZero.h"
#include "GlobalSettings.h"

__global__ void gpuConnectionZeroKernel(double *u_1, double *u1_1, double *u2_1, double *u_2, double *u1_2,
					double *u2_2, double k, int loc1, int loc2, double ls, double sigx,
					double nls, double alpha1, double alpha2)
{
    double w0, w1, w2, w1sq;

    w0 = u_1[loc1] - u_2[loc2];
    w1 = u1_1[loc1] - u1_2[loc2];
    w2 = u2_1[loc1] - u2_2[loc2];
    w1sq = w1 * w1;

    double fac0 = 0.5*ls + (sigx/k) + (0.5*nls*w1sq);
    double fac1 = 1.0 + (fac0 * (alpha1+alpha2));
    double fac2 = -(0.5*ls - (sigx / k) + (0.5*nls*w1sq));

    double force = (1.0 / fac1) * (-fac0*w0 + fac2*w2);

    u_1[loc1] += alpha1 * force;
    u_2[loc2] -= alpha2 * force;
}


GPUConnectionZero::GPUConnectionZero(Component *c1, Component *c2, int loc1, int loc2,
				     double ls, double nls, double t60, double alpha1,
				     double alpha2, double sigx)
{
    this->c1 = c1;
    this->c2 = c2;
    this->loc1 = loc1;
    this->loc2 = loc2;
    linearStiffness = ls;
    nonlinearStiffness = nls;
    this->t60 = t60;
    this->alpha1 = alpha1;
    this->alpha2 = alpha2;
    this->sigx = sigx;
}

GPUConnectionZero::~GPUConnectionZero()
{
}

void GPUConnectionZero::runTimestep(int n)
{
    double k = GlobalSettings::getInstance()->getK();

    if (c2->isOnGPU()) {
	dim3 gridDim(1);
	dim3 blockDim(1);
	// both components are on GPU
	gpuConnectionZeroKernel<<<gridDim, blockDim>>>(c1->getU(), c1->getU1(), c1->getU2(),
						       c2->getU(), c2->getU1(), c2->getU2(),
						       k, loc1, loc2, linearStiffness, sigx,
						       nonlinearStiffness, alpha1, alpha2);
    }
    else {
	double c1u, c1u1, c1u2;
	// only c1 is on GPU - will be slow but should only be used very, very rarely
	// copy values back from it
	cudaMemcpy(&c1u, &c1->getU()[loc1], sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&c1u1, &c1->getU1()[loc1], sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&c1u2, &c1->getU2()[loc1], sizeof(double), cudaMemcpyDeviceToHost);

	double w0 = c1u - c2->getU()[loc2];
	double w1 = c1u1 - c2->getU1()[loc2];
	double w2 = c1u2 - c2->getU2()[loc2];
	double w1sq = w1 * w1;

	double fac0 = 0.5*linearStiffness + (sigx / k) + (0.5*nonlinearStiffness * w1sq);
	double fac1 = 1.0 + (fac0*(alpha1+alpha2));
	double fac2 = -(0.5*linearStiffness - (sigx / k) + (0.5*nonlinearStiffness * w1sq));
	
	double force = (1.0 / fac1) * (-fac0*w0 + fac2*w2);
	
	c2->getU()[loc2] -= alpha2 * force;

	c1u += alpha1 * force;
	cudaMemcpy(&c1->getU()[loc1], &c1u, sizeof(double), cudaMemcpyHostToDevice);
    }
}

