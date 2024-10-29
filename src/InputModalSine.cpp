/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "InputModalSine.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "MathUtil.h"

#include <cstdlib>
#include <cstdio>
using namespace std;


InputModalSine::InputModalSine(ModalPlate *comp, double time, double force,
			       double frequency, double rampUpTime,
			       double steadyTime, double rampDownTime, double x,
			       double y)
    : Input(comp)
{
    int i;

    logMessage(1, "Creating modal sine input: %f, %f, %f, %f, %f, %f, %f, %f", time, force, frequency, rampUpTime, steadyTime, rampDownTime, x, y);

    // convert times to discrete timesteps
    this->startTime = timeToTimestep(time);
    if (this->startTime < firstInputTimestep) firstInputTimestep = this->startTime;

    int rutTs, stTs, rdtTs;

    rutTs = timeToTimestep(rampUpTime);
    stTs = timeToTimestep(steadyTime);
    rdtTs = timeToTimestep(rampDownTime);

    this->duration = rutTs + stTs + rdtTs;

    dat = new double[this->duration];

    // generate window
    for (i = 0; i < rutTs; i++) {
	dat[i] = ((double)i) / ((double)(rutTs-1)) * force;
    }
    for (i = 0; i < stTs; i++) {
	dat[i+rutTs] = force;
    }
    for (i = 0; i < rdtTs; i++) {
	dat[this->duration - 1 - i] = ((double)i) / ((double)(rdtTs-1)) * force;
    }

    double fs = GlobalSettings::getInstance()->getSampleRate();

    // multiply in the sine wave
    for (i = 0; i < this->duration; i++) {
	dat[i] *= sin((frequency * ((double)i)) / fs);
    }

    // generate P
    DIM = comp->getStateSize();
    P = new double[DIM];
    double Lx = comp->getLx();
    double Ly = comp->getLy();
    double h = comp->getH();
    double rho = comp->getRho();
    double *Cnorm = comp->getCnorm();
    double *ov = comp->getOmega();

    x = x * Lx;
    y = y * Ly;

    for (i = 0; i < DIM; i++) {
	P[i] = sin(x * M_PI * ov[(i*3)+1]/Lx) * sin(y * M_PI * ov[(i*3)+2]/Ly);
	P[i] = ((P[i]/rho)/(Lx*Ly/4.0))/h;
	P[i] /= Cnorm[i];
    }

#ifdef USE_GPU
    gpuInputModalSine = NULL;
#endif
}

InputModalSine::~InputModalSine()
{
    delete[] P;
    delete[] dat;
#ifdef USE_GPU
    if (gpuInputModalSine) {
	delete gpuInputModalSine;
    }
#endif
}

void InputModalSine::runTimestep(int n, double *s, double *s1, double *s2)
{
#ifdef USE_GPU
    if (gpuInputModalSine) {
	gpuInputModalSine->runTimestep(n, s);
	return;
    }
#endif

    n -= startTime;
    if ((n >= 0) && (n < duration)) {
	int i;
	for (i = 0; i < DIM; i++) {
	    s[i] += dat[n] * P[i];
	}
    }
}

void InputModalSine::moveToGPU()
{
#ifdef USE_GPU
    logMessage(1, "Moving modal sine input to GPU");
    gpuInputModalSine = new GPUInputModalSine(DIM, dat, P, startTime, duration);
#endif
}

