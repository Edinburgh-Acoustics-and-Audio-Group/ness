/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyAirbox.h"

DummyAirbox::DummyAirbox(string name, double lx, double ly, double lz, double c_a, double rho_a, double tau1)
    : Airbox(name)
{
    setup(lx, ly, lz, c_a, rho_a, tau1);
}

DummyAirbox::~DummyAirbox()
{
}

void DummyAirbox::runTimestep(int n)
{
}

void DummyAirbox::runPartialUpdate(int start, int len)
{
}
