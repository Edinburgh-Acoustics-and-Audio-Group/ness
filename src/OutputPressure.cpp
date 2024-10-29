/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "OutputPressure.h"
#include "GlobalSettings.h"
#include "Logger.h"

OutputPressure::OutputPressure(Component *comp, double pan, double x, double y, double z, int interpolated)
    : Output(comp, pan, x, y, z, interpolated)
{
    SR = GlobalSettings::getInstance()->getSampleRate();
}

OutputPressure::~OutputPressure()
{
}

void OutputPressure::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuOutput) {
	gpuOutput->runTimestep(n);
	return;
    }
#endif

    double *u = component->getU();
    double *u2 = component->getU2();
    double uval, u2val;

    switch (interp.type) {
    case INTERPOLATION_NONE:
	uval = u[index];
	u2val = u2[index];
	break;
    case INTERPOLATION_LINEAR:
	uval = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1];
	u2val = interp.alpha[0] * u2[index] +
	    interp.alpha[1] * u2[index+1];
	break;
    case INTERPOLATION_BILINEAR:
	uval = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1] +
	    interp.alpha[2] * u[index+interp.nx] +
	    interp.alpha[3] * u[index+interp.nx+1];
	u2val = interp.alpha[0] * u2[index] +
	    interp.alpha[1] * u2[index+1] +
	    interp.alpha[2] * u2[index+interp.nx] +
	    interp.alpha[3] * u2[index+interp.nx+1];
	break;
    case INTERPOLATION_TRILINEAR:
	uval = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1] +
	    interp.alpha[2] * u[index+interp.nx] +
	    interp.alpha[3] * u[index+interp.nx+1] +
	    interp.alpha[4] * u[index+interp.nxny] +
	    interp.alpha[5] * u[index+interp.nxny+1] +
	    interp.alpha[6] * u[index+interp.nxny+interp.nx] +
	    interp.alpha[7] * u[index+interp.nxny+interp.nx+1];
	u2val = interp.alpha[0] * u2[index] +
	    interp.alpha[1] * u2[index+1] +
	    interp.alpha[2] * u2[index+interp.nx] +
	    interp.alpha[3] * u2[index+interp.nx+1] +
	    interp.alpha[4] * u2[index+interp.nxny] +
	    interp.alpha[5] * u2[index+interp.nxny+1] +
	    interp.alpha[6] * u2[index+interp.nxny+interp.nx] +
	    interp.alpha[7] * u2[index+interp.nxny+interp.nx+1];
	break;
    }
    data[n] = SR * (uval - u2val);
}

void OutputPressure::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (component->isOnGPU()) {
	logMessage(1, "Moving output to GPU");
	gpuOutput = new GPUOutput(GPUOUTPUTTYPE_PRESSURE, component, index, &interp);
    }
#endif
}
