
/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy subclass of Component2D for testing
 */

#ifndef _DUMMYCOMPONENT2D_H_
#define _DUMMYCOMPONENT2D_H_

#include "Component2D.h"

class DummyComponent2D : public Component2D {
 public:
    DummyComponent2D(string name, int nx, int ny);
    virtual ~DummyComponent2D();

    virtual void runTimestep(int n);

    vector<Input*> *getInputs() { return &inputs; }

    virtual bool isOnGPU();
    virtual bool moveToGPU();
    
 protected:
    bool onGPU;
};

#endif
