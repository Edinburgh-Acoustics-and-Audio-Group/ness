/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "InputPluck.h"
#include "Logger.h"
#include "MathUtil.h"

#include <cmath>
using namespace std;

InputPluck::InputPluck(Component *comp, double x, double y, double z, double startTime,
		       double duration, double amplitude, int interpolated, int negated)
    : InputSample(comp, comp->getAlpha(), x, y, z, interpolated, negated)
{
    // convert times to discrete timesteps
    this->startTime = timeToTimestep(startTime);
    this->duration = timeToTimestep(duration) + 1;

    if (this->startTime < firstInputTimestep) firstInputTimestep = this->startTime;

    // generate pluck data
    data = new double[this->duration];
    int i;
    for (i = 0; i < this->duration; i++) {
	data[i] = 0.5 * amplitude * (1.0 - cos(M_PI * ((double)i) / ((double)this->duration)));
    }

    logMessage(1, "Pluck on %s, start=%d, length=%d, position=%d", comp->getName().c_str(),
	       this->startTime, this->duration, index);
}


InputPluck::~InputPluck()
{
}
