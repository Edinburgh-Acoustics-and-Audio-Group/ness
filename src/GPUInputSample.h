/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU-accelerated sample input
 */

#ifndef _GPU_INPUT_SAMPLE_H_
#define _GPU_INPUT_SAMPLE_H_

#include "Component.h"

class GPUInputSample {
 public:
    GPUInputSample(double *data, int len, double multiplier, int index, InterpolationInfo *interp);
    virtual ~GPUInputSample();

    void runTimestep(int n, double *s);

 protected:
    double *d_data;
    int index;
    double multiplier;
    InterpolationInfo *interp;
};

#endif
