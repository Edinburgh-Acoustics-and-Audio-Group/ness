/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * String component
 */
#ifndef _COMPONENTSTRING_H_
#define _COMPONENTSTRING_H_

#include "Component1D.h"

extern "C" {
#include "csrmatrix.h"
};

class ComponentString : public Component1D {
 public:
    ComponentString(string name, double L, double rho, double T, double E, double r,
		    double T60_0, double T60_1000, double xc1, double yc1, double xc2,
		    double yc2);
    virtual ~ComponentString();

    virtual void runTimestep(int n);

    double getRho() { return rho; }
    double getSig0() { return sig0; }
    double getH() { return h; }

    double getXc1() { return xc1; }
    double getYc1() { return yc1; }
    double getXc2() { return xc2; }
    double getYc2() { return yc2; }

    void getConnectionValues(double *l);
    void setConnectionValues(double *l);

    virtual void logMatrices();

 protected:
    double L;
    double rho;
    double T;
    double E;
    double radius;
    double T60_0;
    double T60_1000;

    double k;
    double h;
    double sig0;

    // position if attached to soundboard
    double xc1, yc1, xc2, yc2;

    // for connections
    double isb0, isb1, isb2;
    double isc0, isc1;
    double js0;

    CSRmatrix *B, *C;
};

#endif
