/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Lip input for brass instruments
 */

#ifndef _INPUTLIPS_H_
#define _INPUTLIPS_H_

#include "Input.h"
#include "BrassInstrument.h"
#include "BreakpointFunction.h"

#include <vector>
using namespace std;

class InputLips : public Input {
 public:
    InputLips(Component *comp, vector<double> &Sr, vector<double> &mu, vector<double> &sigma, vector<double> &H,
	      vector<double> &w, vector<double> &pressure, vector<double> &lip_frequency, vector<double> &vibamp,
	      vector<double> &vibfreq, vector<double> &tremamp, vector<double> &tremfreq, vector<double> &noiseamp);
    virtual ~InputLips();

    virtual void runTimestep(int n, double *s, double *s1, double *s2);
    virtual void moveToGPU();

 protected:
    double getRand();

    BrassInstrument *brass;
    
    double y, y1, y2;
    double b2, b3, c1, c2, d1, d2;
    double dp, sqrtdp;
    double Smain0;
    double rho;

    BreakpointFunction *bf_Sr, *bf_mu, *bf_sigma, *bf_H, *bf_wint;
    BreakpointFunction *bf_tremamp, *bf_tremfreq, *bf_noise, *bf_pm;
    BreakpointFunction *bf_vibamp, *bf_vibfreq, *bf_omega;
};

#endif
