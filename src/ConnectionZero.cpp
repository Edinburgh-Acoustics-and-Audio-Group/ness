/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "ConnectionZero.h"
#include "GlobalSettings.h"
#include "Logger.h"

#include <cmath>
using namespace std;

ConnectionZero::ConnectionZero(Component *c1, Component *c2, double xi1, double yi1,
			       double zi1, double xi2, double yi2, double zi2,
			       double ls, double nls, double t60i)
    : ConnectionP2P(c1, c2, xi1, yi1, zi1, xi2, yi2, zi2)
{
    GlobalSettings *gs = GlobalSettings::getInstance();

    linearStiffness = ls;
    nonlinearStiffness = nls;
    t60 = t60i;

    // pre-compute what we can in here
    // we need k (from GlobalSettings), and alpha values from the components
    double k = gs->getK();

    alpha1 = c1->getAlpha();
    alpha2 = c2->getAlpha();

    sigx = (6.0 * log(10.0)) / t60;
    if (!gs->getLossMode()) {
	// FIXME: handle all loss modes properly
	sigx = 0.0;
    }

    logMessage(1, "Connection locations: %d, %d, alphas: %.20f, %.20f, sig: %.20f", loc1, loc2, alpha1,
	       alpha2, sigx);

    energy = NULL;
    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps()];
    }

#ifdef USE_GPU
    gpuConnectionZero = NULL;
#endif
}

ConnectionZero::~ConnectionZero()
{
    if (energy) delete[] energy;
#ifdef USE_GPU
    if (gpuConnectionZero) delete gpuConnectionZero;
#endif
}

void ConnectionZero::runTimestep(int n)
{
    double w0, w1, w2, w1sq;

#ifdef USE_GPU
    if (gpuConnectionZero) {
	gpuConnectionZero->runTimestep(n);
	return;
    }
#endif

    double k = GlobalSettings::getInstance()->getK();

    w0 = c1->getU()[loc1] - c2->getU()[loc2];
    w1 = c1->getU1()[loc1] - c2->getU1()[loc2];
    w2 = c1->getU2()[loc1] - c2->getU2()[loc2];
    w1sq = w1 * w1;

    double fac0 = 0.5*linearStiffness + (sigx / k) + (0.5*nonlinearStiffness * w1sq);
    double fac1 = 1.0 + (fac0*(alpha1+alpha2));
    double fac2 = -(0.5*linearStiffness - (sigx / k) + (0.5*nonlinearStiffness * w1sq));

    double force = (1.0 / fac1) * (-fac0*w0 + fac2*w2);

    c1->getU()[loc1] += alpha1 * force;
    c2->getU()[loc2] -= alpha2 * force;

    if (energy) {
	w0 = c1->getU()[loc1] - c2->getU()[loc2];
	energy[n] = (0.25 * linearStiffness * ((w0*w0) + (w1sq))) +
	    (0.25 * nonlinearStiffness * (w0*w0) * (w1sq));
    }
}

void ConnectionZero::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (c1->isOnGPU()) {
	// c1 is on GPU - we move to GPU (whether c2 is there or not)
	logMessage(1, "Moving connection to GPU");
	gpuConnectionZero = new GPUConnectionZero(c1, c2, loc1, loc2, linearStiffness,
						  nonlinearStiffness, t60, alpha1, alpha2,
						  sigx);
    }
    else if (c2->isOnGPU()) {
	// c1 is not on GPU but c2 is - we swap the components and move them to GPU
	gpuConnectionZero = new GPUConnectionZero(c2, c1, loc2, loc1, linearStiffness,
						  nonlinearStiffness, t60, alpha2, alpha1,
						  sigx);
    }
#endif
}

double *ConnectionZero::getEnergy()
{
    return energy;
}
