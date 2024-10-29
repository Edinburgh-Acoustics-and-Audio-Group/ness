/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Abstract superclass for all airbox components
 */
#ifndef _AIRBOX_H_
#define _AIRBOX_H_

#include "Component3D.h"

class Airbox : public Component3D {
 public:
    Airbox(string name);
    virtual ~Airbox();

    virtual void runTimestep(int n) = 0;

    virtual int getGPUScore();

    virtual void runPartialUpdate(int start, int len) = 0;

    double getLX() { return LX; }
    double getLY() { return LY; }
    double getLZ() { return LZ; }
    double getQ() { return Q; }
    double getGamma() { return gammabar; }
    double getLambda() { return lambda; }
    double getK() { return k; }
    double getRhoA() { return rho_a; }

    virtual void addPlate(int zb, double *true_Psi);

 protected:
    void setup(double lx, double ly, double lz, double c_a, double rho_a, double tau1 = 0.0);

    // physical dimensions in metres
    double LX, LY, LZ;

    double k, Q, Gamma, rho_a, c_a;

    // viscosity variables
    double gammabar, lambda, tau1;
};

#endif
