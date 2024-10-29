/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * This is the abstract superclass for all 1D components (bars, strings,
 * tubes, etc.)
 */
#ifndef _COMPONENT1D_H_
#define _COMPONENT1D_H_

#include "Component.h"

class Component1D : public Component {
 public:
    Component1D(string name);
    virtual ~Component1D();

    virtual int getIndexf(double x, double y=0.0, double z=0.0);
    virtual int getIndex(int x, int y=0, int z=0);
    virtual void getInterpolationInfo(InterpolationInfo *info, double x, double y=0.0, double z=0.0);

    virtual void runTimestep(int n) = 0;

 protected:
    // allocate and initialise state arrays of the correct size
    void allocateState(int n);
};

#endif
