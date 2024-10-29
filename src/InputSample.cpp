/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "InputSample.h"
#include "Logger.h"
#include "SettingsManager.h"

// interpolated and negated parameters:
// 0 = no, 1 = yes, -1 = fallback to component or global setting
InputSample::InputSample(Component *comp, double mult, double x, double y, double z, int interpolated, int negated)
    : Input(comp, x, y, z)
{
    int i;
    data = NULL;
    multiplier = mult;

    SettingsManager *sm = SettingsManager::getInstance();

    if (negated == 1) {
	multiplier = -multiplier;
    }
    else if (negated < 0) {
	if (sm->getBoolSetting(comp->getName(), "negate_inputs")) {
	    multiplier = -multiplier;
	}
    }

    // set up for interpolation if we have it
    if (interpolated == 0) {
	interp.type = INTERPOLATION_NONE;
    }
    else if (interpolated == 1) {
	comp->getInterpolationInfo(&interp, x, y, z);
    }
    else {
	// fallback to global setting
	if (!sm->getBoolSetting(comp->getName(), "interpolate_inputs")) {
	    interp.type = INTERPOLATION_NONE;
	}
	else {
	    comp->getInterpolationInfo(&interp, x, y, z);
	}
    }

    // data, startTime and duration must be setup by subclass

#ifdef USE_GPU
    gpuInputSample = NULL;
#endif
}

InputSample::~InputSample()
{
    if (data) delete[] data;

#ifdef USE_GPU
    if (gpuInputSample) {
	delete gpuInputSample;
    }
#endif
}

void InputSample::runTimestep(int n, double *s, double *s1, double *s2)
{
    n -= startTime;
    if ((n >= 0) && (n < duration)) {
#ifdef USE_GPU
	if (gpuInputSample) {
	    gpuInputSample->runTimestep(n, s);
	    return;
	}
#endif
	switch (interp.type) {
	case INTERPOLATION_NONE:
	    s[index] += multiplier * data[n];
	    break;
	case INTERPOLATION_LINEAR:
	    s[index] += multiplier * interp.alpha[0] * data[n];
	    s[index+1] += multiplier * interp.alpha[1] * data[n];
	    break;
	case INTERPOLATION_BILINEAR:
	    s[index] += multiplier * interp.alpha[0] * data[n];
	    s[index+1] += multiplier * interp.alpha[1] * data[n];
	    s[index+interp.nx] += multiplier * interp.alpha[2] * data[n];
	    s[index+interp.nx+1] += multiplier * interp.alpha[3] * data[n];
	    break;
	case INTERPOLATION_TRILINEAR:
	    s[index] += multiplier * interp.alpha[0] * data[n];
	    s[index+1] += multiplier * interp.alpha[1] * data[n];
	    s[index+interp.nx] += multiplier * interp.alpha[2] * data[n];
	    s[index+interp.nx+1] += multiplier * interp.alpha[3] * data[n];
	    s[index+interp.nxny] += multiplier * interp.alpha[4] * data[n];
	    s[index+interp.nxny+1] += multiplier * interp.alpha[5] * data[n];
	    s[index+interp.nxny+interp.nx] += multiplier * interp.alpha[6] * data[n];
	    s[index+interp.nxny+interp.nx+1] += multiplier * interp.alpha[7] * data[n];
	    break;
	}
    }
}

void InputSample::moveToGPU()
{
#ifdef USE_GPU
    logMessage(1, "Moving sample input to GPU");
    gpuInputSample = new GPUInputSample(data, duration, multiplier, index, &interp);
#endif
}
