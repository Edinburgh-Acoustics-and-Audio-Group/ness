/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Represents a sine wave input for a modal plate
 */
#ifndef _INPUTMODALSINE_H_
#define _INPUTMODALSINE_H_

#include "Input.h"
#include "ModalPlate.h"

#ifdef USE_GPU
#include "GPUInputModalSine.h"
#endif

class InputModalSine : public Input {
 public:
    InputModalSine(ModalPlate *comp, double time, double force, double frequency,
		   double rampUpTime, double steadyTime, double rampDownTime,
		   double x, double y);
    virtual ~InputModalSine();

    virtual void runTimestep(int n, double *s, double *s1, double *s2);
    virtual void moveToGPU();

 protected:
    double *P;
    int DIM;
    double *dat;

#ifdef USE_GPU
    GPUInputModalSine *gpuInputModalSine;
#endif
};

#endif
