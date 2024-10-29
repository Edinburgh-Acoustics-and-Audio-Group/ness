/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "InputLips.h"
#include "MathUtil.h"
#include "GlobalSettings.h"
#include "Logger.h"

#include <cmath>
#include <cstdlib>
using namespace std;


InputLips::InputLips(Component *comp, vector<double> &Sr, vector<double> &mu, vector<double> &sigma,
		     vector<double> &H, vector<double> &w, vector<double> &pressure,
		     vector<double> &lip_frequency, vector<double> &vibamp, vector<double> &vibfreq,
		     vector<double> &tremamp, vector<double> &tremfreq, vector<double> &noiseamp)
    : Input(comp, 0.0)
{
    logMessage(1, "Entering InputLips constructor");

    double kk = GlobalSettings::getInstance()->getK();

    brass = dynamic_cast<BrassInstrument*>(comp);
    if (brass == NULL) {
	logMessage(5, "Lips input only works with brass instruments");
	exit(1);
    }

    dp = 0.0;
    y = 0.0;
    y1 = 0.0;
    y2 = 0.0;

    // lip effective surface area
    bf_Sr = new BreakpointFunction(Sr.data(), Sr.size() / 2, kk);

    // lip effective mass
    bf_mu = new BreakpointFunction(mu.data(), mu.size() / 2, kk);

    // lip damping
    bf_sigma = new BreakpointFunction(sigma.data(), sigma.size() / 2, kk);

    // equilibrium separation
    bf_H = new BreakpointFunction(H.data(), H.size() / 2, kk);

    // width
    bf_wint = new BreakpointFunction(w.data(), w.size() / 2, kk);

    // tremolo
    bf_tremamp = new BreakpointFunction(tremamp.data(), tremamp.size() / 2, kk);
    bf_tremfreq = new BreakpointFunction(tremfreq.data(), tremfreq.size() / 2, kk);

    // noise
    bf_noise = new BreakpointFunction(noiseamp.data(), noiseamp.size() / 2, kk);

    // mouth pressure
    bf_pm = new BreakpointFunction(pressure.data(), pressure.size() / 2, kk);

    // vibrato
    bf_vibamp = new BreakpointFunction(vibamp.data(), vibamp.size() / 2, kk);
    bf_vibfreq = new BreakpointFunction(vibfreq.data(), vibfreq.size() / 2, kk);

    // lip angular frequency with vibrato added
    bf_omega = new BreakpointFunction(lip_frequency.data(), lip_frequency.size() / 2, kk);
    
    c1 = brass->getC1();
    rho = brass->getRho();
    Smain0 = brass->getSmain0();

    setFirstInputTimestep(0);
}

InputLips::~InputLips()
{
    delete bf_Sr;
    delete bf_mu;
    delete bf_sigma;
    delete bf_H;
    delete bf_wint;
    delete bf_tremamp;
    delete bf_tremfreq;
    delete bf_noise;
    delete bf_pm;
    delete bf_vibamp;
    delete bf_vibfreq;
    delete bf_omega;
}

double InputLips::getRand()
{
    return ((double)rand()) / ((double)RAND_MAX);
}

/*
 * Normally when this is called for an Input, s, s1, and s2 correspond to u, u1, and u2.
 * For the lips this is not the case. They are:
 *  - s = pmain
 *  - s1 = pmain1
 *  - s2 = vmain1
 */
void InputLips::runTimestep(int n, double *s, double *s1, double *s2)
{
    double kk = GlobalSettings::getInstance()->getK();

    // compute the breakpoint functions
    double H = bf_H->getValue();
    double lipa = bf_wint->getValue() * sqrt(2.0 / rho);
    double sigmaint = bf_sigma->getValue();

    double vibamp = bf_vibamp->getValue();
    double vibfreq = bf_vibfreq->getValue();
    double vibrato = 1.0 + vibamp * sin(2.0 * M_PI * vibfreq * (((double)n)*kk));
    double omega = 2.0 * M_PI * bf_omega->getValue() * vibrato;

    double Srint = bf_Sr->getValue();
    double muint = bf_mu->getValue();

    double a1, a3, b1, al;
    al = 1.0 / (1.0 / (kk*kk) + sigmaint * 0.5 / kk + 0.5 * omega * omega);
    double bl = al * (2.0 / (kk*kk));
    double cl = al * (sigmaint * 0.5 / kk - 1.0 / (kk*kk) - 0.5 * omega * omega);
    double dl = al * (Srint / muint);

    a1 = 2.0 / kk + sigmaint + kk * omega * omega;
    a3 = Srint / muint;
    b1 = Srint * a3 / a1;
    double b1c1 = 1.0 / (-(b1 + c1));
    double om2 = -omega * omega * Srint / a1;
    double k2 = 0.5 * kk * kk * a1 / Srint;

    double tremamp = bf_tremamp->getValue();
    double tremfreq = bf_tremfreq->getValue();
    double trem = 1.0 + tremamp * sin(2.0 * M_PI * tremfreq * (((double)n)*kk));
    double noise = 2.0 * bf_noise->getValue() * (getRand() - 0.5);
    double pm = bf_pm->getValue() * (trem * (1.0 + noise));

    bf_H->next();
    bf_wint->next();
    bf_sigma->next();
    bf_omega->next();
    bf_vibamp->next();
    bf_vibfreq->next();
    bf_Sr->next();
    bf_mu->next();
    bf_tremamp->next();
    bf_tremfreq->next();
    bf_noise->next();
    bf_pm->next();

    /* pressure difference update */
    b2 = (y1 + H + fabs(y1 + H)) * lipa;
    b3 = ((y1 - y2) / k2) + (om2 * y2);
    c2 = c1 * (pm - s1[0]) + Smain0*s2[0];
    d1 = b2 * b1c1;
    d2 = (b3 - c2) * b1c1;
    sqrtdp = 0.5 * (d1 + sqrt(d1*d1 + 4.0*fabs(d2)));
    dp = sqrtdp * sqrtdp;
    if (d2 < 0.0) dp = -dp;
    
    /* lip update */
    y = bl*y1 + cl*y2 + dl*dp;
    y2 = y1;
    y1 = y;

    /* mouthpiece input */
    s[0] = 2.0 * (pm - dp) - s1[0];
}

void InputLips::moveToGPU()
{
#ifdef USE_GPU
    // not supported on GPU, which is fine since BrassInstrument isn't
    logMessage(5, "ERROR: attempting to move lips input to GPU!");
#endif
}
