/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy subclass of Component1D for testing
 */

#ifndef _DUMMYCOMPONENT1D_H_
#define _DUMMYCOMPONENT1D_H_

#include "Component1D.h"

class DummyComponent1D : public Component1D {
 public:
    DummyComponent1D(string name, int n);
    virtual ~DummyComponent1D();

    virtual void runTimestep(int n);

    vector<Input*> *getInputs() { return &inputs; }

    virtual bool isOnGPU();
    virtual bool moveToGPU();

 protected:
    bool onGPU;
};

#endif
