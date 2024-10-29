/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy subclass of ConnectionP2P for testing
 */

#ifndef _DUMMYCONNECTIONP2P_H_
#define _DUMMYCONNECTIONP2P_H_

#include "ConnectionP2P.h"

class DummyConnectionP2P : public ConnectionP2P {
 public:
    DummyConnectionP2P(Component *c1, Component *c2, double x1, double x2);
    virtual ~DummyConnectionP2P();
    virtual void runTimestep(int n);
};

#endif
