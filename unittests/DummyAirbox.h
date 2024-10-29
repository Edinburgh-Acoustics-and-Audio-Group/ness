/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy airbox for testing
 */

#ifndef _DUMMYAIRBOX_H_
#define _DUMMYAIRBOX_H_

#include "Airbox.h"

class DummyAirbox : public Airbox {
 public:
    DummyAirbox(string name, double lx, double ly, double lz, double c_a, double rho_a, double tau1);
    virtual ~DummyAirbox();

    virtual void runTimestep(int n);
    virtual void runPartialUpdate(int start, int len);
};

#endif
