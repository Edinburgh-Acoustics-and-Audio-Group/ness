/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Represents a pressure output from a component.
 */
#ifndef _OUTPUT_PRESSURE_H_
#define _OUTPUT_PRESSURE_H_

#include "Output.h"

class OutputPressure : public Output {
 public:
    OutputPressure(Component *comp, double pan, double x, double y = 0.0, double z = 0.0, int interpolated = -1);
    virtual ~OutputPressure();

    virtual void runTimestep(int n);
    virtual void maybeMoveToGPU();

 protected:
    // sample rate
    double SR;
};

#endif
