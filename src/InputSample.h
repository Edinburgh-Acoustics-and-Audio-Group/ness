/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * An input that plays from a sample in a buffer - could be a WAV
 * file, or something that's computed at startup such as a strike, a
 * pluck, etc. Subclasses will take care of the differences between
 * these.
 */
#ifndef _INPUT_SAMPLE_H_
#define _INPUT_SAMPLE_H_

#include "Input.h"

#ifdef USE_GPU
#include "GPUInputSample.h"
#endif

class InputSample : public Input {
 public:
    InputSample(Component *comp, double mult, double x, double y=0, double z=0,
		int interpolated = -1, int negated = -1);
    virtual ~InputSample();

    virtual void runTimestep(int n, double *s, double *s1, double *s2);

    virtual void moveToGPU();

 protected:
    // data to be input at each timestep
    double *data;

    // scaling factor
    double multiplier;

    InterpolationInfo interp;

#ifdef USE_GPU
    GPUInputSample *gpuInputSample;
#endif
};

#endif
