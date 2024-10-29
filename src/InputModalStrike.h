/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Represents a strike on a modal plate
 */

#ifndef _INPUTMODALSTRIKE_H_
#define _INPUTMODALSTRIKE_H_

#include "Input.h"
#include "ModalPlate.h"

#ifdef USE_GPU
#include "GPUInputModalStrike.h"
#endif

class InputModalStrike : public Input {
 public:
    InputModalStrike(ModalPlate *comp, double x, double y, double startTime,
		     double duration, double amplitude);
    virtual ~InputModalStrike();
    
    virtual void runTimestep(int n, double *s, double *s1, double *s2);
    virtual void moveToGPU();

 protected:
    double *P;
    int DIM;
    double amplitude;
    double T0, Twid;

#ifdef USE_GPU
    GPUInputModalStrike *gpuInputModalStrike;
#endif
};


#endif
