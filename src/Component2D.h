/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * This is the abstract superclass for all 2D components (plates, boards,
 * membranes, etc.)
 */
#ifndef _COMPONENT2D_H_
#define _COMPONENT2D_H_

#include "Component.h"

class Component2D : public Component {
 public:
    Component2D(string name);
    virtual ~Component2D();

    virtual int getIndexf(double x, double y=0.0, double z=0.0);
    virtual int getIndex(int x, int y=0, int z=0);
    virtual void getInterpolationInfo(InterpolationInfo *info, double x, double y=0.0, double z=0.0);
    virtual void runTimestep(int n) = 0;

    int getNx() { return nx; }
    int getNy() { return ny; }

 protected:
    // allocate and initialise state arrays
    void allocateState(int nx, int ny);

    // grid size in each dimension
    int nx, ny;
};

#endif
