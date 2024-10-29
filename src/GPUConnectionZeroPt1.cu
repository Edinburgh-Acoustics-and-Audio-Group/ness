/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "GPUConnectionZeroPt1.h"
#include "MathUtil.h"

#include <cmath>
using namespace std;

#define EPSILON 2.2204460492503131e-16

__global__ void gpuConnectionZeroPt1Kernel(double *u_1, double *u1_1, double *u2_1,
					   double *u_2, double *u1_2, double *u2_2,
					   int loc1, int loc2, double alpha1, double alpha2,
					   double K, double alpha, double offset,
					   double one_sided)
{
    double w0 = u_1[loc1] - u_2[loc2];
    double w1 = u1_1[loc1] - u1_2[loc2];
    double w2 = u2_1[loc1] - u2_2[loc2];

    double a =  0.5 * (w1 + w2);
    double b = -0.5 * (w0 - w2);
    double M =  0.5 * (alpha1 + alpha2);

    // newton solver
    double coeff = K / (alpha + 1.0);
    double r = 1.0;
    double R, F, temp;

    int nn;
    for (nn = 0; nn < 5; nn++) {
	double ra = r + a;
	double rae = fabs(ra) - offset;
	double ae = fabs(a) - offset;

	double sra = copysign(1.0, ra);
	double srae = copysign(1.0, rae);
	double sae = copysign(1.0, ae);
	double sa = copysign(1.0, a);
	double sr = copysign(1.0, r);

	double phi_ra = 0.5 * coeff * (1.0 - 0.5 * one_sided * (1.0 - sra)) * (1.0 + srae)
	    * pow(fabs(rae), alpha + 1.0);
	double phi_a = 0.5 * coeff * (1.0 - 0.5 * one_sided * (1.0 - sa)) * (1.0 + sae)
	    * pow(fabs(ae), alpha + 1.0);
	double phi_prime = 0.5 * K * sra * (1.0 - 0.5 * one_sided * (1.0 - sra))
	    * (1.0 + srae) * pow(fabs(rae), alpha);
	if (fabs(r) > EPSILON) {
	    R = sr * (phi_ra - phi_a) / fabs(r);
	    F = r + M*R + b;
	    temp = (r * phi_prime - phi_ra + phi_a) / (r*r);
	}
	else {
	    R = 0.5 * (K * (1.0 - 0.5 * one_sided * (1.0 - sae))) * (1.0 + sae)
		* pow(fabs(ae), alpha);
	    F = r + M*R + b;
	    temp = 0.5 * (0.5 * alpha * (alpha+1.0) * K
			  * (1.0 - 0.5 * one_sided * (1.0 - sae)) * (1.0 + sae)
			  * pow(fabs(ae), (alpha - 1.0)));
	}
	r = r - F / (1.0 + M * temp);
    }

    // done newton, result is in -R
    u_1[loc1] += alpha1 * (-R);
    u_2[loc2] -= alpha2 * (-R);
}

GPUConnectionZeroPt1::GPUConnectionZeroPt1(Component *c1, Component *c2, int loc1, int loc2, double alpha1,
					   double alpha2, double K, double alpha, double offset,
					   double one_sided)
{
    this->c1 = c1;
    this->c2 = c2;
    this->loc1 = loc1;
    this->loc2 = loc2;
    this->alpha1 = alpha1;
    this->alpha2 = alpha2;
    this->K = K;
    this->alpha = alpha;
    this->offset = offset;
    this->one_sided = one_sided;
}

GPUConnectionZeroPt1::~GPUConnectionZeroPt1()
{
}

void GPUConnectionZeroPt1::runTimestep(int n)
{
    if (c2->isOnGPU()) {
	// both components are on GPU
	dim3 dimGrid(1);
	dim3 dimBlock(1);
	gpuConnectionZeroPt1Kernel<<<dimGrid, dimBlock>>>(c1->getU(), c1->getU1(), c1->getU2(),
							  c2->getU(), c2->getU1(), c2->getU2(),
							  loc1, loc2, alpha1, alpha2, K, alpha, offset,
							  one_sided);
    }
    else {
	// only c1 is on GPU. need to do slow data copies
	double c1u, c1u1, c1u2;
	cudaMemcpy(&c1u, &c1->getU()[loc1], sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&c1u1, &c1->getU1()[loc1], sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&c1u2, &c1->getU2()[loc1], sizeof(double), cudaMemcpyDeviceToHost);

	double w0 = c1u - c2->getU()[loc2];
	double w1 = c1u1 - c2->getU1()[loc2];
	double w2 = c1u2 - c2->getU2()[loc2];

	double a =  0.5 * (w1 + w2);
	double b = -0.5 * (w0 - w2);
	double M =  0.5 * (alpha1 + alpha2);

	double force = newtonSolver(a, b, M, K, alpha, offset, one_sided, NULL);

	c2->getU()[loc2] -= alpha2 * force;

	c1u += alpha1 * force;
	cudaMemcpy(&c1->getU()[loc1], &c1u, sizeof(double), cudaMemcpyHostToDevice);
    }
}
