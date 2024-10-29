/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * A bow input
 */

#ifndef _INPUTBOW_H_
#define _INPUTBOW_H_

#include "Input.h"

#ifdef USE_GPU
#include "GPUInputBow.h"
#endif

class InputBow : public Input {
 public:
    InputBow(Component *comp, double x, double y, double z, double startTime, double duration,
	     double fAmp, double vAmp, double fric, double rampTime);
    virtual ~InputBow();

    virtual void runTimestep(int n, double *s, double *s1, double *s2);
    virtual void moveToGPU();

 protected:
    double *velocity;
    double *force;
    double jb;
    double sigma;
    double k;
    int itnum;

#ifdef USE_GPU
    GPUInputBow *gpuInputBow;
#endif
};

#endif
