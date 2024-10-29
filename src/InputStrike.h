/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * An input that consists of a sinusoidal strike on a component
 */

#ifndef _INPUTSTRIKE_H_
#define _INPUTSTRIKE_H_

#include "InputSample.h"

class InputStrike : public InputSample {
 public:
    InputStrike(Component *comp, double x, double y, double z, double startTime,
		double duration, double amplitude, int interpolated = -1, int negated = -1);
    virtual ~InputStrike();
};

#endif
