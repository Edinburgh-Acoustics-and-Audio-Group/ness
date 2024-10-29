/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Represents a Zero code-style connection
 */
#ifndef _CONNECTIONZERO_H_
#define _CONNECTIONZERO_H_

#include "ConnectionP2P.h"

#ifdef USE_GPU
#include "GPUConnectionZero.h"
#endif

class ConnectionZero : public ConnectionP2P {
 public:
    ConnectionZero(Component *c1, Component *c2, double xi1, double yi1, double zi1,
		   double xi2, double yi2, double zi2,
		   double ls, double nls, double t60i);
    virtual ~ConnectionZero();

    virtual void runTimestep(int n);

    virtual void maybeMoveToGPU();

    virtual double *getEnergy();

 protected:
    double linearStiffness;
    double nonlinearStiffness;
    double t60;

    double alpha1, alpha2, sigx;

    double *energy;

#ifdef USE_GPU
    GPUConnectionZero *gpuConnectionZero;
#endif
};

#endif
