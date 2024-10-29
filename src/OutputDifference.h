/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Represents a difference output from a component.
 */
#ifndef _OUTPUT_DIFFERENCE_H_
#define _OUTPUT_DIFFERENCE_H_

#include "Output.h"

class OutputDifference : public Output {
 public:
    OutputDifference(Component *comp, double pan, double x, double y = 0.0, double z = 0.0, int interpolated = -1);
    virtual ~OutputDifference();

    virtual void runTimestep(int n);
    virtual void maybeMoveToGPU();

 protected:
    // sample rate
    double SR;
};

#endif
