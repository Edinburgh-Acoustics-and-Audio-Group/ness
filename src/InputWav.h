/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Input from a WAV file
 */
#ifndef _INPUTWAV_H_
#define _INPUTWAV_H_

#include "InputSample.h"

#include <string>
using namespace std;

class InputWav : public InputSample {
 public:
    InputWav(Component *comp, double x, double y, double z, string filename, double startTime,
	     double gain, int interpolated = -1, int negated = -1);
    virtual ~InputWav();
};

#endif
