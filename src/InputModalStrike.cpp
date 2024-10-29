/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "InputModalStrike.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "MathUtil.h"

InputModalStrike::InputModalStrike(ModalPlate *comp, double x, double y,
				   double startTime, double duration,
				   double amplitude)
    : Input(comp)
{
    logMessage(1, "Creating modal strike: %f, %f, %f, %f, %f", x, y, startTime,
	       duration, amplitude);

    // convert times to discrete timesteps
    this->startTime = timeToTimestep(startTime);
    this->duration = timeToTimestep(duration);

    if (this->startTime < firstInputTimestep) firstInputTimestep = this->startTime;

    this->amplitude = amplitude;

    this->Twid = duration / 2.0;
    this->T0 = startTime + Twid;

    // generate P
    DIM = comp->getStateSize();
    P = new double[DIM];
    double Lx = comp->getLx();
    double Ly = comp->getLy();
    double h = comp->getH();
    double rho = comp->getRho();
    double *Cnorm = comp->getCnorm();
    double *ov = comp->getOmega();
    int i;

    x = x * Lx;
    y = y * Ly;

    for (i = 0; i < DIM; i++) {
	P[i] = sin(x * M_PI * ov[(i*3)+1]/Lx) * sin(y * M_PI * ov[(i*3)+2]/Ly);
	P[i] = ((P[i]/rho)/(Lx*Ly/4.0))/h;
	P[i] /= Cnorm[i];
	//logMessage(1, "P[i]=%.20f", P[i]);
    }

#ifdef USE_GPU
    gpuInputModalStrike = NULL;
#endif
}

InputModalStrike::~InputModalStrike()
{
    delete[] P;
#ifdef USE_GPU
    if (gpuInputModalStrike) {
	delete gpuInputModalStrike;
    }
#endif
}

void InputModalStrike::runTimestep(int n, double *s, double *s1, double *s2)
{
#ifdef USE_GPU
    if (gpuInputModalStrike) {
	gpuInputModalStrike->runTimestep(n, s);
	return;
    }
#endif

    if ((n >= startTime) && (n < (startTime+duration))) {
	double dn = (double)n;
	double fs = GlobalSettings::getInstance()->getSampleRate();
	double val = 0.5 * amplitude * (1.0 + cos(M_PI * (dn/fs/Twid - T0/Twid)));

	// add it to component's modes
	int i;
	for (i = 0; i < DIM; i++) {
	    s[i] += val * P[i];
	}
    }
}

void InputModalStrike::moveToGPU()
{
#ifdef USE_GPU
    logMessage(1, "Moving modal strike input to GPU");
    gpuInputModalStrike = new GPUInputModalStrike(P, startTime, duration,
						  amplitude, DIM, T0, Twid);
#endif
}

