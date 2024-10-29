/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "InputBow.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "MathUtil.h"

#include <cmath>
using namespace std;

InputBow::InputBow(Component *comp, double x, double y, double z, double startTime,
		   double duration, double fAmp, double vAmp, double fric, double rampTime)
    : Input(comp, x, y, z)
{
    this->startTime = timeToTimestep(startTime);
    this->duration = timeToTimestep(duration);

    if (this->startTime < firstInputTimestep) firstInputTimestep = this->startTime;

    GlobalSettings *gs = GlobalSettings::getInstance();
    double sr = gs->getSampleRate();

    int ramp_ind = (int)floor(rampTime * sr);
    sigma = fric;

    velocity = new double[this->duration + 1];
    force = new double[this->duration + 1];

    int i;
    for (i = 0; i < ramp_ind; i++) {
	force[i] = 0.5 * fAmp * (1.0 - cos(M_PI * ((double)i) / ((double)ramp_ind)));
	velocity[i] = 0.5 * vAmp * (1.0 - cos(M_PI * ((double)i) / ((double)ramp_ind)));
	
	force[i + this->duration - ramp_ind + 1] = 0.5 * fAmp *
	    (1.0 + cos(M_PI * ((double)i) / ((double)ramp_ind)));
	velocity[i + this->duration - ramp_ind + 1] = 0.5 * vAmp *
	    (1.0 + cos(M_PI * ((double)i) / ((double)ramp_ind)));	
    }
    for (i = ramp_ind; i <= (this->duration - ramp_ind); i++) {
	force[i] = fAmp;
	velocity[i] = vAmp;
    }
    // FIXME: should this have a ramp down at the end??? It's missing from Zero!
    
    jb = comp->getBowFactor();
    k = gs->getK();
    itnum = 10;

#ifdef USE_GPU
    gpuInputBow = NULL;
#endif
}

InputBow::~InputBow()
{
    delete[] velocity;
    delete[] force;

#ifdef USE_GPU
    if (gpuInputBow) delete gpuInputBow;
#endif
}

void InputBow::runTimestep(int n, double *s, double *s1, double *s2)
{
    n -= startTime;
    if ((n < 0) || (n > duration)) return;

#ifdef USE_GPU
    if (gpuInputBow) {
	gpuInputBow->runTimestep(n, s, s2);
	return;
    }
#endif

    int i;
    double g = (1.0 / (2.0 * k)) * (s[index] - s2[index]);
    double alpha = 0.0;
    double vfac = 0.0;

    g -= velocity[n];
    alpha = jb * force[n];
    vfac = velocity[n];

    // This could be optimised so that the update isn't done when the bow's not
    // doing anything. Have tested and it appears to give the same result
    double v = 0;
    for (i = 0; i < itnum; i++) {
	v = v - (v - g + sqrt(2.0 * exp(1.0)) * alpha * sigma * v *
		 exp(-(sigma*sigma)*(v*v))) / (1.0 + sqrt(2.0*exp(1.0))* alpha * sigma *
					       (1.0 - 2.0*(sigma*sigma)*(v*v)) *
					       exp(-(sigma*sigma)*(v*v)));
    }
    s[index] = s2[index] + 2.0*k*(v + vfac);
}

void InputBow::moveToGPU()
{
#ifdef USE_GPU
    logMessage(1, "Moving bow input to GPU");
    gpuInputBow = new GPUInputBow(duration, velocity, force, k, jb, itnum, sigma, index);
#endif
}
