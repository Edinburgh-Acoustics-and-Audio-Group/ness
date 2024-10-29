/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "OutputDifference.h"
#include "GlobalSettings.h"
#include "Logger.h"

OutputDifference::OutputDifference(Component *comp, double pan, double x, double y, double z, int interpolated)
    : Output(comp, pan, x, y, z, interpolated)
{
    SR = GlobalSettings::getInstance()->getSampleRate();
}

OutputDifference::~OutputDifference()
{
}

void OutputDifference::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuOutput) {
	gpuOutput->runTimestep(n);
	return;
    }
#endif
    double *u = component->getU();
    double *u1 = component->getU1();
    double uval, u1val;

    switch (interp.type) {
    case INTERPOLATION_NONE:
	uval = u[index];
	u1val = u1[index];
	break;
    case INTERPOLATION_LINEAR:
	uval = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1];
	u1val = interp.alpha[0] * u1[index] +
	    interp.alpha[1] * u1[index+1];
	break;
    case INTERPOLATION_BILINEAR:
	uval = interp.alpha[0] * u[index] +
	    interp.alpha[1] * u[index+1] +
	    interp.alpha[2] * u[index+interp.nx] +
	    interp.alpha[3] * u[index+interp.nx+1];
	u1val = interp.alpha[0] * u1[index] +
	    interp.alpha[1] * u1[index+1] +
	    interp.alpha[2] * u1[index+interp.nx] +
	    interp.alpha[3] * u1[index+interp.nx+1];
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
	u1val = interp.alpha[0] * u1[index] +
	    interp.alpha[1] * u1[index+1] +
	    interp.alpha[2] * u1[index+interp.nx] +
	    interp.alpha[3] * u1[index+interp.nx+1] +
	    interp.alpha[4] * u1[index+interp.nxny] +
	    interp.alpha[5] * u1[index+interp.nxny+1] +
	    interp.alpha[6] * u1[index+interp.nxny+interp.nx] +
	    interp.alpha[7] * u1[index+interp.nxny+interp.nx+1];
	break;
    }
    data[n] = SR * (uval - u1val);
}

void OutputDifference::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (component->isOnGPU()) {
	logMessage(1, "Moving output to GPU");
	gpuOutput = new GPUOutput(GPUOUTPUTTYPE_DIFFERENCE, component, index, &interp);
    }
#endif
}
