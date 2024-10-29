/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "InputValve.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "BrassInstrument.h"
#include "MathUtil.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;

InputValve::InputValve(Component *comp, int valveIdx, vector<double> &opening,
		       vector<double> &vibratoFrequency, vector<double> &vibratoAmplitude)
    : Input(comp, 0.0)
{
    double kk = GlobalSettings::getInstance()->getK();

    BrassInstrument *brass = dynamic_cast<BrassInstrument*>(comp);
    if (brass == NULL) {
	logMessage(5, "Error: valve inputs can only be used on brass instruments");
	exit(1);
    }

    bp_opening = new BreakpointFunction(opening.data(), opening.size() / 2, kk);
    bp_vibamp = new BreakpointFunction(vibratoAmplitude.data(), vibratoAmplitude.size() / 2, kk);
    bp_vibfreq = new BreakpointFunction(vibratoFrequency.data(), vibratoFrequency.size() / 2, kk);

    brass->setValveData(valveIdx, this);

    setFirstInputTimestep(0);

    ts = 0;
}

InputValve::~InputValve()
{
    delete bp_opening;
    delete bp_vibamp;
    delete bp_vibfreq;
}

void InputValve::runTimestep(int n, double *s, double *s1, double *s2)
{
    // all done at startup, nothing to do here
}

void InputValve::moveToGPU()
{
    // brass not supported on GPU yet, so nothing to do
}

double InputValve::nextQd()
{
    double kk = GlobalSettings::getInstance()->getK();
    double qd = bp_opening->getValue() + (bp_vibamp->getValue() *
					  sin(2.0 * M_PI * bp_vibfreq->getValue() * (((double)ts)*kk)));
    if (qd < 0.0) qd = 0.0;
    if (qd > 1.0) qd = 1.0;

    bp_opening->next();
    bp_vibamp->next();
    bp_vibfreq->next();

    ts++;

    return qd;
}
