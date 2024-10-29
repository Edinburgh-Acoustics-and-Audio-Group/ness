/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * GPU-accelerated sine input for modal plate
 */

#ifndef _GPU_INPUT_MODAL_SINE_H_
#define _GPU_INPUT_MODAL_SINE_H_

#include "ModalPlate.h"

class GPUInputModalSine {
 public:
    GPUInputModalSine(int DIM, double *dat, double *P, int startTime, int duration);
    virtual ~GPUInputModalSine();

    void runTimestep(int n, double *s);

 protected:
    int DIM;
    int startTime;
    int duration;

    double *d_dat;
    double *d_P;
};

#endif
