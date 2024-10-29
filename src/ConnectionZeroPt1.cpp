/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "ConnectionZeroPt1.h"

#include "GlobalSettings.h"
#include "Logger.h"
#include "MathUtil.h"

#include <cmath>
#include <cfloat>
using namespace std;

ConnectionZeroPt1::ConnectionZeroPt1(Component *c1, Component *c2, double xi1, double yi1,
				     double zi1, double xi2, double yi2, double zi2, double K,
				     double alpha, double one_sided, double offset)
    : ConnectionP2P(c1, c2, xi1, yi1, zi1, xi2, yi2, zi2)
{
    GlobalSettings *gs = GlobalSettings::getInstance();

    this->K = K;
    this->alpha = alpha;
    this->one_sided = one_sided;
    this->offset = offset;

    alpha1 = c1->getAlpha();
    alpha2 = c2->getAlpha();

    energy = NULL;
    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps()];
    }

#ifdef USE_GPU
    gpuConnectionZeroPt1 = NULL;
#endif
}

ConnectionZeroPt1::~ConnectionZeroPt1()
{
    if (energy) delete[] energy;
#ifdef USE_GPU
    if (gpuConnectionZeroPt1) delete gpuConnectionZeroPt1;
#endif
}

void ConnectionZeroPt1::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuConnectionZeroPt1) {
	gpuConnectionZeroPt1->runTimestep(n);
	return;
    }
#endif

    double w0, w1, w2;
    double phi_ra;

    w0 = c1->getU()[loc1] - c2->getU()[loc2];
    w1 = c1->getU1()[loc1] - c2->getU1()[loc2];
    w2 = c1->getU2()[loc1] - c2->getU2()[loc2];

    double a =  0.5 * (w1 + w2);
    double b = -0.5 * (w0 - w2);
    double M =  0.5 * (alpha1 + alpha2);

    double force = newtonSolver(a, b, M, K, alpha, offset, one_sided, &phi_ra);
    //double force = 0.0;

    c1->getU()[loc1] += alpha1 * force;
    c2->getU()[loc2] -= alpha2 * force;

    if (energy) {
	/*w0 = c1->getU()[loc1] - c2->getU()[loc2];
	double wdel = 0.5 * (w0 + w1);
	double wds = (wdel < 0.0) ? -1.0 : 1.0;
	energy[n] = ((K / (alpha+1.0)) * (1.0 - 0.5*one_sided*(1.0-wds)) * pow((fabs(wdel)), (alpha+1.0)));*/
	energy[n] = phi_ra;
    }
}

void ConnectionZeroPt1::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (c1->isOnGPU()) {
	// c1 is on the GPU - we're moving there, whether c2 is there or not
	gpuConnectionZeroPt1 = new GPUConnectionZeroPt1(c1, c2, loc1, loc2, alpha1, alpha2, K,
							alpha, offset, one_sided);
    }
    else if (c2->isOnGPU()) {
	// c2 is on the GPU but not c1. Move there, but swap the components
	gpuConnectionZeroPt1 = new GPUConnectionZeroPt1(c2, c1, loc2, loc1, alpha2, alpha1, K,
							alpha, offset, one_sided);
    }
#endif
}

double *ConnectionZeroPt1::getEnergy()
{
    return energy;
}
