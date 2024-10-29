/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * This is the abstract superclass for all 3D components (airboxes,
 * whole rooms, etc.)
 */
#ifndef _COMPONENT3D_H_
#define _COMPONENT3D_H_

#include "Component.h"

class Component3D : public Component {
 public:
    Component3D(string name);
    virtual ~Component3D();

    virtual void runTimestep(int n) = 0;
    virtual int getIndexf(double x, double y=0.0, double z=0.0);
    virtual int getIndex(int x, int y=0, int z=0);
    virtual void getInterpolationInfo(InterpolationInfo *info, double x, double y=0.0, double z=0.0);

    int getNx() { return nx; }
    int getNy() { return ny; }
    int getNz() { return nz; }

 protected:
    // grid size in each dimension
    int nx, ny, nz;

    // grid points in X-Y plane
    int nxny;

    // allocate and initialise state arrays
    void allocateState(int nx, int ny, int nz);
};

#endif
