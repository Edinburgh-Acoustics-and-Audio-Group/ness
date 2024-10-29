/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "Output.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "SettingsManager.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
using namespace std;

// interpolated: 0 = no, 1 = yes, -1 = fall back to component's default or
// global default
Output::Output(Component *comp, double pan, double x, double y, double z,
	       int interpolated)
{
    component = comp;
    this->pan = pan;
    index = comp->getIndexf(x, y, z);

    if (interpolated == 0) {
	// interpolation disabled
	interp.type = INTERPOLATION_NONE;
    }
    else if (interpolated == 1) {
	// interpolation enabled
	comp->getInterpolationInfo(&interp, x, y, z);
    } else {
	// fall back to component or global interpolation value
	if (!SettingsManager::getInstance()->getBoolSetting(comp->getName(), "interpolate_outputs")) {
	    interp.type = INTERPOLATION_NONE;
	}
	else {
	    comp->getInterpolationInfo(&interp, x, y, z);
	}
    }

    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    data = new double[NF];

    memset(data, 0, NF * sizeof(double));

#ifdef USE_GPU
    gpuOutput = NULL;
#endif

    logMessage(1, "Adding output from %s at position %d, pan %f", comp->getName().c_str(),
	       index, pan);
}

Output::Output()
{
    data = NULL;
}

Output::~Output()
{
    if (data) delete[] data;

#ifdef USE_GPU
    delete gpuOutput;
#endif
}

void Output::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuOutput) {
	gpuOutput->runTimestep(n);
	return;
    }
#endif

    double *u = component->getU();

    switch (interp.type) {
    case INTERPOLATION_NONE:
	data[n] = u[index];
	break;
    case INTERPOLATION_LINEAR:
	data[n] = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1];
	break;
    case INTERPOLATION_BILINEAR:
	data[n] = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1] +
	    interp.alpha[2] * u[index+interp.nx] +
	    interp.alpha[3] * u[index+interp.nx+1];
	break;
    case INTERPOLATION_TRILINEAR:
	data[n] = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1] +
	    interp.alpha[2] * u[index+interp.nx] +
	    interp.alpha[3] * u[index+interp.nx+1] +
	    interp.alpha[4] * u[index+interp.nxny] +
	    interp.alpha[5] * u[index+interp.nxny+1] +
	    interp.alpha[6] * u[index+interp.nxny+interp.nx] +
	    interp.alpha[7] * u[index+interp.nxny+interp.nx+1];
	break;
    }
}

void Output::saveRawData(string filename)
{
    ofstream of(filename.c_str(), ios::out | ios::binary);
    if (!of.good()) {
	logMessage(5, "Failed to create raw output file %s", filename.c_str());
	return;
    }
    of.write((const char *)data,
	     GlobalSettings::getInstance()->getNumTimesteps() * sizeof(double));
    of.close();
}

void Output::highPassFilter()
{
    int i;
    for (i = GlobalSettings::getInstance()->getNumTimesteps()-1; i > 0; i--) {
	data[i] = data[i] - data[i-1];
    }
    data[i] = 0.0;
}

double Output::getMaxValue()
{
    int i;
    double max = 0.0;
    for (i = 0; i < GlobalSettings::getInstance()->getNumTimesteps(); i++) {
	double v = fabs(data[i]);
	if (v > max) max = v;
    }
    return max;
}

void Output::normalise()
{
    double max = getMaxValue();

    if (max == 0.0) return;

    int i;
    for (i = 0; i < GlobalSettings::getInstance()->getNumTimesteps(); i++) {
	data[i] /= max;
    }
}

void Output::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (component->isOnGPU()) {
	logMessage(1, "Moving output to GPU");
	gpuOutput = new GPUOutput(GPUOUTPUTTYPE_NORMAL, component, index, &interp);
    }
#endif
}

void Output::copyFromGPU()
{
#ifdef USE_GPU
    if (gpuOutput) {
	logMessage(1, "Copying data back from GPU output");
	gpuOutput->getData(data);
    }
#endif
}
