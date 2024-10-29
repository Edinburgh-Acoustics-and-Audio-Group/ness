/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU-accelerated ZeroPt1 code style connection
 */

#ifndef _GPU_CONNECTION_ZEROPT1_H_
#define _GPU_CONNECTION_ZEROPT1_H_

#include "Component.h"

class GPUConnectionZeroPt1 {
 public:
    GPUConnectionZeroPt1(Component *c1, Component *c2, int loc1, int loc2, double alpha1,
			 double alpha2, double K, double alpha, double offset, double one_sided);
    virtual ~GPUConnectionZeroPt1();

    virtual void runTimestep(int n);

 protected:
    Component *c1, *c2;
    int loc1, loc2;
    double alpha1, alpha2;
    double K, alpha, offset;
    double one_sided;
};

#endif
