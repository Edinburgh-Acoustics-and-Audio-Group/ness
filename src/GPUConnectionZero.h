/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU-accelerated Zero code style connection
 */

#ifndef _GPU_CONNECTION_ZERO_H_
#define _GPU_CONNECTION_ZERO_H_

#include "Component.h"

class GPUConnectionZero {
 public:
    GPUConnectionZero(Component *c1, Component *c2, int loc1, int loc2, double ls, double nls,
		      double t60, double alpha1, double alpha2, double sigx);
    virtual ~GPUConnectionZero();

    virtual void runTimestep(int n);

 protected:
    Component *c1, *c2;
    int loc1, loc2;

    double linearStiffness;
    double nonlinearStiffness;
    double t60;

    double alpha1, alpha2, sigx;
};

#endif
