/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy InputSample subclass used for testing
 */

#ifndef _DUMMYINPUTSAMPLE_H_
#define _DUMMYINPUTSAMPLE_H_

#include "InputSample.h"

class DummyInputSample : public InputSample {
 public:
    DummyInputSample(Component *comp, double mult, double x, double y, double z, double *data, int datalen, int interpolated = -1, int negated = -1);
    virtual ~DummyInputSample();
};

#endif
