/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyConnectionP2P.h"
#include "Logger.h"

DummyConnectionP2P::DummyConnectionP2P(Component *c1, Component *c2, double x1, double x2)
  : ConnectionP2P(c1, c2, x1, 0.0, 0.0, x2, 0.0, 0.0)
{
}

DummyConnectionP2P::~DummyConnectionP2P()
{
}

void DummyConnectionP2P::runTimestep(int n)
{
}
