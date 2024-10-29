/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * An input that consists of a half-sinusoidal pluck of a component
 */

#ifndef _INPUTPLUCK_H_
#define _INPUTPLUCK_H_

#include "InputSample.h"

class InputPluck : public InputSample {
 public:
    InputPluck(Component *comp, double x, double y, double z, double startTime,
	       double duration, double amplitude, int interpolated = -1, int negated = -1);
    virtual ~InputPluck();
};

#endif
