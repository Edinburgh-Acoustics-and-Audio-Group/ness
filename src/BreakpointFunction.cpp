/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "BreakpointFunction.h"

#include <iostream>
using namespace std;

BreakpointFunction::BreakpointFunction(double *x, double *v, int lenx, double k)
{
    int i;
    this->lenx = lenx;
    this->x = new double[lenx];
    this->v = new double[lenx];
    this->k = k;
    for (i = 0; i < lenx; i++) {
	this->x[i] = x[i];
	this->v[i] = v[i];
    }
    t = 0.0;
    nextSegment();
}

BreakpointFunction::BreakpointFunction(double *xandv, int lenx, double k)
{
    int i;
    this->lenx = lenx;
    this->x = new double[lenx];
    this->v = new double[lenx];
    this->k = k;
    for (i = 0; i < lenx; i++) {
	this->x[i] = xandv[i*2];
	this->v[i] = xandv[i*2+1];
    }
    t = 0.0;
    nextSegment();    
}

BreakpointFunction::~BreakpointFunction()
{
    delete[] x;
    delete[] v;
}

void BreakpointFunction::nextSegment()
{
    int i;

    // find out where we are at t
    if (t < x[0]) {
	// far LHS
	// nothing happens until we actually reach x[0]
	value = v[0];
	nexttime = x[0];
	increment = 0.0;
    }
    else if (t >= x[lenx-1]) {
	// far RHS
	// stay at same level for rest of simulation
	value = v[lenx-1];
	nexttime = 1e30;
	increment = 0.0;
    }
    else {
	// in the middle
	// work out which points we're in between
	for (i = 0; i < (lenx - 1); i++) {
	    if ((x[i] <= t) && (x[i+1] > t)) {
		double alpha = (t - x[i]) / (x[i+1] - x[i]);

		// work out current value
		value = (1.0 - alpha) * v[i] + alpha * v[i+1];

		// store next transition time
		nexttime = x[i+1];

		// work out the increment
		alpha = ((t+k) - x[i]) / (x[i+1] - x[i]);
		double nextval = (1.0 - alpha) * v[i] + alpha * v[i+1];
		increment = nextval - value;
		break;
	    }
	}
    }
}

void BreakpointFunction::next()
{
    t += k;
    value += increment;
    if (t >= nexttime) {
	// work out new value, increment and nexttime
	nextSegment();
    }
}
