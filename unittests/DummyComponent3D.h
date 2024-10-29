/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy subclass of Component3D for testing
 */

#ifndef _DUMMYCOMPONENT3D_H_
#define _DUMMYCOMPONENT3D_H_

#include "Component3D.h"

class DummyComponent3D : public Component3D {
 public:
    DummyComponent3D(string name, int nx, int ny, int nz);
    virtual ~DummyComponent3D();

    virtual void runTimestep(int n);

    virtual bool isOnGPU();
    virtual bool moveToGPU();

 protected:
    bool onGPU;
};

#endif
