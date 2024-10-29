/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Implementation of linear breakpoint functions (similar to Matlab's interp1)
 */

#ifndef _BREAKPOINTFUNCTION_H_
#define _BREAKPOINTFUNCTION_H_

class BreakpointFunction {
 public:
    BreakpointFunction(double *x, double *v, int lenx, double k);
    BreakpointFunction(double *xandv, int lenx, double k);
    virtual ~BreakpointFunction();

    /* return the current value of the function */
    double getValue() {
	return value;
    }

    /* move onto the next timestep */
    void next();

 protected:
    void nextSegment();

    double value;
    double increment;

    double t;
    double nexttime;

    double *x, *v;
    int lenx;
    double k;
};

#endif
