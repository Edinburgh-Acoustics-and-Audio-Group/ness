/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Zero point 1-style connection
 */
#ifndef _CONNECTIONZEROPT1_H_
#define _CONNECTIONZEROPT1_H_

#include "ConnectionP2P.h"

#ifdef USE_GPU
#include "GPUConnectionZeroPt1.h"
#endif

class ConnectionZeroPt1 : public ConnectionP2P {
 public:
    ConnectionZeroPt1(Component *c1, Component *c2, double xi1, double yi1, double zi1,
		      double xi2, double yi2, double zi2, double K, double alpha,
		      double one_sided, double offset);
    virtual ~ConnectionZeroPt1();

    virtual void runTimestep(int n);

    virtual void maybeMoveToGPU();

    virtual double *getEnergy();

 protected:
    double K, alpha, one_sided, offset;
    double alpha1, alpha2;

    double *energy;

#ifdef USE_GPU
    GPUConnectionZeroPt1 *gpuConnectionZeroPt1;
#endif
};

#endif
