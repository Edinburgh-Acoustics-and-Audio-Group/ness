/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * GPU-accelerated sample input for modal plate
 */

#ifndef _GPU_INPUT_MODAL_STRIKE_H_
#define _GPU_INPUT_MODAL_STRIKE_H_

#include "ModalPlate.h"

class GPUInputModalStrike {
 public:
    GPUInputModalStrike(double *P, int startTime, int duration, double amplitude,
			int DIM, double T0, double Twid);
    virtual ~GPUInputModalStrike();

    void runTimestep(int n, double *s);

 protected:
    double *d_p;
    int DIM;
    double amplitude;
    int startTime;
    int duration;
    double T0, Twid;
};


#endif

