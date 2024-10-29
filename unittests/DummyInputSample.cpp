/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyInputSample.h"
#include "GlobalSettings.h"

#include <cstring>
using namespace std;

DummyInputSample::DummyInputSample(Component *comp, double mult, double x, double y, double z, double *data, int datalen, int interpolated, int negated)
    : InputSample(comp, mult, x, y, z, interpolated, negated)
{
    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    this->data = new double[NF];
    memset(this->data, 0, NF * sizeof(double));

    if (datalen > NF) datalen = NF;
    memcpy(this->data, data, datalen * sizeof(double));

    startTime = 0;
    duration = datalen;
}

DummyInputSample::~DummyInputSample()
{
}
