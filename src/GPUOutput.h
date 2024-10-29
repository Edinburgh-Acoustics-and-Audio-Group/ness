/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU-accelerated output
 */

#ifndef _GPU_OUTPUT_H_
#define _GPU_OUTPUT_H_

#include "Component.h"

enum { GPUOUTPUTTYPE_NORMAL, GPUOUTPUTTYPE_DIFFERENCE, GPUOUTPUTTYPE_PRESSURE };

class GPUOutput {
 public:
    GPUOutput(int type, Component *comp, int idx, InterpolationInfo *interp);
    GPUOutput(Component *comp);
    virtual ~GPUOutput();

    virtual void runTimestep(int n);
    virtual void getData(double *buf);

 protected:
    int type;
    Component *component;
    int idx;

    double *d_data;

    InterpolationInfo *interp;
};

#endif
