/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Represents a valve input on a wind instrument.
 */

#ifndef _INPUTVALVE_H_
#define _INPUTVALVE_H_

#include "Input.h"
#include "BreakpointFunction.h"

#include <vector>
using namespace std;

class InputValve : public Input {
 public:
    InputValve(Component *comp, int valveIdx, vector<double> &opening,
	       vector<double> &vibratoFrequency, vector<double> &vibratoAmplitude);
    virtual ~InputValve();

    virtual void runTimestep(int n, double *s, double *s1, double *s2);
    virtual void moveToGPU();

    double nextQd();

    virtual bool preventsEnergyConservation() {
	return false;
    }

 protected:

    BreakpointFunction *bp_opening, *bp_vibamp, *bp_vibfreq;

    int ts;
};

#endif
