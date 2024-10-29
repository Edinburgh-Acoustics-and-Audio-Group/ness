/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "OutputModal.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "MathUtil.h"

#ifdef USE_GPU
#include "GPUOutputModal.h"
#endif

#include <cmath>
using namespace std;

OutputModal::OutputModal(ModalPlate *comp, double pan, double x, double y)
{
    logMessage(1, "Creating modal output from %s: %f,%f", comp->getName().c_str(),
	       x, y);

    component = comp;
    this->pan = pan;

    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    data = new double[NF];

    memset(data, 0, NF * sizeof(double));

    // calculate read point vector
    DIM = comp->getStateSize();
    h = comp->getH();
    rp = new double[DIM];
    double *ov = comp->getOmega();
    double Lx = comp->getLx();
    double Ly = comp->getLy();
    double op1 = x * Lx;
    double op2 = y * Ly;
    int i;
    for (i = 0; i < DIM; i++) {
	rp[i] = sin(ov[(i*3)+1] * M_PI * op1 / Lx) * sin(ov[(i*3)+2] * M_PI * op2 / Ly);
    }

    logMessage(1, "Adding modal output from %s at position %f,%f, pan %f", comp->getName().c_str(),
	       x, y, pan);

#ifdef USE_GPU
    gpuOutput = NULL;
#endif
}

OutputModal::~OutputModal()
{
    delete[] rp;
}

void OutputModal::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuOutput) {
	gpuOutput->runTimestep(n);
	return;
    }
#endif
    double *q = component->getU();
    int i;
    double val = 0.0;
    for (i = 0; i < DIM; i++) {
	val += rp[i] * q[i];
    }
    data[n] = val / h;
}

void OutputModal::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (component->isOnGPU()) {
	logMessage(1, "Moving modal output to GPU");
	ModalPlate *mp = dynamic_cast<ModalPlate*>(component);
	gpuOutput = new GPUOutputModal(mp, rp);
    }
#endif
}
