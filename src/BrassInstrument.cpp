/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "BrassInstrument.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "SettingsManager.h"
#include "Input.h"
#include "InputValve.h"
#include "MathUtil.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
using namespace std;

// viscotherm types
enum { VTT_TUSTIN, VTT_AL, VTT_GL };


BrassInstrument::BrassInstrument(string name, double temperature, int Nvalve, vector<double> &vpos,
				 vector<double> &vdl, vector<double> &vbl, vector<double> &bore)
    : Component1D(name)
{
    init(temperature, Nvalve, vpos, vdl, vbl, &bore);
}

vector<double> *BrassInstrument::customBore(double xm, vector<double> &x0, double L, double rm,
					    vector<double> &r0, double rb, double fp)
{
    double sumx0 = 0.0;
    int i, j;
    for (i = 0; i < x0.size(); i++) {
	sumx0 += x0[i];
    }
    double xb = L - sumx0 - xm;
    double h = 1.0;
    double val;

    vector<double> *bore = new vector<double>();

    // generate mouthpiece bore
    double x = 0.0;
    double r;
    while (x <= xm) {
	r = 0.5 * (rm - r0[0]) * (1.0 + cos(M_PI * x / xm)) + r0[0];
	bore->push_back(x);
	bore->push_back(r);
	x += h;
    }

    // generate middle of bore
    double dummy = xm;
    for (i = 0; i < x0.size(); i++) {
	switch ((int)r0[(i*3)+2]) {
	case 1:
	    x = dummy + x0[i];
	    r = r0[(i*3)+1];
	    bore->push_back(x);
	    bore->push_back(r);
	    break;
	case 2:
	    x = dummy + h;
	    val = h;
	    while (x <= (dummy + x0[i])) {
		r = r0[(i*3)] + (r0[(i*3)+1] - r0[(i*3)]) *
		    (sin((M_PI * val) / x0[i]) * sin((M_PI * val) / x0[i]));
		bore->push_back(x);
		bore->push_back(r);
		x += h;
		val += h;
	    }
	    break;
	case 3:
	    x = dummy + h;
	    val = h;
	    while (x <= (dummy + x0[i])) {
		r = 0.5 * (r0[(i*3)+1] - r0[(i*3)]) * (1.0 - cos((M_PI * val) / x0[i])) + r0[(i*3)];
		bore->push_back(x);
		bore->push_back(r);
		x += h;
		val += h;
	    }
	    break;
	}
	dummy = dummy + x0[i];
    }

    // generate flare
    double rdummy = r;
    x = dummy + h;
    val = h;
    while (x <= (dummy+xb)) {
	r = (rb - rdummy) * pow((val / xb), fp) + rdummy;
	bore->push_back(x);
	bore->push_back(r);
	x += h;
	val += h;
    }
    
    return bore;
}


// constructor for use with custom bores
BrassInstrument::BrassInstrument(string name, double temperature, int Nvalve, vector<double> &vpos,
				 vector<double> &vdl, vector<double> &vbl, double xm, vector<double> &x0,
				 double L, double rm, vector<double> &r0, double rb, double fp)
    : Component1D(name)
{
    vector<double> *bore = customBore(xm, x0, L, rm, r0, rb, fp);
    init(temperature, Nvalve, vpos, vdl, vbl, bore);
    delete bore;
}

void BrassInstrument::init(double temperature, int Nvalve, vector<double> &vpos,
			   vector<double> &vdl, vector<double> &vbl, vector<double> *bore)
{
    double L;
    double *Lmain;

    double h;
    int *Nmain, *Nby;
    double *xmain, *xd;
    int d0, main0;
    int Nmaintotal, Ndtotal, Nbytotal;
    double lmsum, ldsum;
    double *r, *x;
    int xlen;
    int nmainsum, ndsum, nbysum;

    double *Smainorig, *Sdorig;
    int smonext, sdonext, end;
    double *hmainvec;

    double R1, R2, Cr, Lr;

    int i, j, k;

    logMessage(1, "Entering BrassInstrument initialiser");

    startupProfiler.start(0);

    SettingsManager *sm = SettingsManager::getInstance();

    int loss = sm->getIntSetting(name, "loss_mode");

    this->Nvalve = Nvalve;
    historySize = 19;

    valvesSet = 0;

    vr = 0.0;
    pr = 0.0;

    GlobalSettings *gs = GlobalSettings::getInstance();
    double FS = gs->getSampleRate();
    double kk = gs->getK();
    int K = gs->getNumTimesteps();

    /* compute thermodynamic constants for given air temperature */
    deltaT = temperature - 26.85;
    rho = 1.1769 * (1.0 - (0.00335 * deltaT));
    c = 347.23 * (1.0 + (0.00166 * deltaT));
    eta = 1.846e-5 * (1.0 + (0.0025 * deltaT));
    nu = 0.841 * (1.0 - (0.0002 * deltaT));
    gamma = 1.4017 * (1.0 - (0.00002 * deltaT));
    
    if (loss == 0) {
	eta = 0.0;
	historySize = 1;
    }

    /* convert bore data from mm to m and diameter to radius */
    int boresize = bore->size() / 2;
    for (i = 0; i < boresize; i++) {
	(*bore)[i*2] = (*bore)[i*2] / 1000.0;
	(*bore)[i*2+1] = (*bore)[i*2+1] / 2000.0;
    }

    /* convert valve data from mm to m */
    for (i = 0; i < Nvalve; i++) {
	vpos[i] = vpos[i] / 1000.0;
	vdl[i] = vdl[i] / 1000.0;
	vbl[i] = vbl[i] / 1000.0;
    }

    /* instrument length, valve and bypass lengths */
    /* just use vdl and vbl directly as Ld and Lby */
    L = (*bore)[(boresize-1)*2];

    logMessage(1, "Computing lengths");
    
    /* get length of main tubing */
    Lmain = new double[Nvalve+1];
    if (Nvalve == 1) {
	Lmain[0] = vpos[0];
	Lmain[1] = L - vpos[0] - vdl[0];
    }
    else {
	Lmain[0] = vpos[0];
	for (i = 0; i < (Nvalve-1); i++) {
	    Lmain[i+1] = vpos[i+1] - vpos[i] -vdl[i];
	}
	Lmain[Nvalve] = L - vpos[Nvalve-1] - vdl[Nvalve-1];
    }

    /* general length step */
    h = c * kk;

    /* work out vector lengths */
    Nmain = new int[Nvalve+1];
    Nd = new int[Nvalve];
    Nby = new int[Nvalve];
    Nmaintotal = 0;
    Ndtotal = 0;
    Nbytotal = 0;
    for (i = 0; i < (Nvalve+1); i++) {
	Nmain[i] = (int)floor(Lmain[i] / h);
	Nmaintotal += Nmain[i];
    }
    for (i = 0; i < Nvalve; i++) {
	Nd[i] = (int)floor(vdl[i] / h);
	Nby[i] = (int)floor((vbl[i] + vdl[i] + vdl[i]) / h);
	Ndtotal += Nd[i];
	Nbytotal += Nby[i];
    }

    mainSize = Nmaintotal + Nvalve + 1;
    defaultSize = Ndtotal + Nvalve;
    bypassSize = Nbytotal + Nvalve;

    /* work out length step for each section */
    hmain = new double[Nvalve+1];
    hd = new double[Nvalve];
    hby = new double[Nvalve];
    for (i = 0; i < (Nvalve+1); i++) {
	hmain[i] = Lmain[i] / (double)Nmain[i];
    }
    for (i = 0; i < Nvalve; i++) {
	hd[i] = vdl[i] / (double)Nd[i];
	hby[i] = (vbl[i] + vdl[i] + vdl[i]) / (double)Nby[i];
    }

    logMessage(1, "Computing x positions and bore");

    /* work out x positions of each element of main and default tubes */
    xmain = new double[Nmaintotal];
    xd = new double[Ndtotal];
    for (i = 0; i < Nmain[0]; i++) {
	xmain[i] = (Lmain[0] * (0.5 + (double)i)) / (double)Nmain[0];
    }
    d0 = 0;
    main0 = Nmain[0];
    lmsum = 0.0;
    ldsum = 0.0;
    for (i = 0; i < Nvalve; i++) {
	lmsum += Lmain[i];
	ldsum += vdl[i];

	for (j = 0; j < Nd[i]; j++) {
	    xd[d0] = vpos[i] + ((vdl[i] * (0.5 + (double)j)) / (double)Nd[i]);
	    d0++;
	}
	for (j = 0; j < Nmain[i+1]; j++) {
	    xmain[main0] = lmsum + ldsum + ((Lmain[i+1] * (0.5 + (double)j)) / (double)Nmain[i+1]);
	    main0++;
	}
    }
    
    /* bore stuff */
    xlen = (int)floor(L / (h / 2.0)) + 1;
    x = new double[xlen];
    r = new double[xlen];
    for (i = 0; i < xlen; i++) {
	x[i] = (h / 2) * (double)i;
    }
    interp1_interleaved(bore->data(), x, boresize, xlen, r);

    /* get indices of junctions in the different tubes */
    logMessage(1, "Getting junction indices");
    mjin = new int[Nvalve+2];
    mjout = new int[Nvalve+1];
    djin = new int[Nvalve];
    djout = new int[Nvalve];
    byjin = new int[Nvalve];
    byjout = new int[Nvalve];
    nmainsum = 0;
    ndsum = 0;
    nbysum = 0;
    mjin[0] = 0;
    for (i = 0; i < Nvalve; i++) {
	nmainsum += (Nmain[i]+1);
	ndsum += (Nd[i]+1);
	nbysum += (Nby[i]+1);
	mjout[i] = nmainsum - 1;
	mjin[i+1] = nmainsum;
	djout[i] = ndsum - 1;
	byjout[i] = nbysum - 1;
    }
    mjout[Nvalve] = Nmaintotal + Nvalve;
    djin[0] = 0;
    byjin[0] = 0;
    for (i = 1; i < Nvalve; i++) {
	djin[i] = djout[i-1] + 1;
	byjin[i] = byjout[i-1] + 1;
    }

    startupProfiler.end(0);

    /* get cross sectional area of all tubes */
    startupProfiler.start(1);
    logMessage(1, "Getting cross sectional area of tubes");
    Smain = new double[mainSize];
    Smainbar = new double[mainSize];
    Smainorig = new double[mainSize];

    Sdorig = new double[defaultSize];
    Sdti = new double[defaultSize];
    Sdbarti = new double[defaultSize];
    Sbyti = new double[bypassSize];
    Sbybarti = new double[bypassSize];

    tempd1 = new double[defaultSize];
    tempdbar1 = new double[defaultSize];
    tempby1 = new double[bypassSize];
    tempbybar1 = new double[bypassSize];

    Sd1 = new double[defaultSize];
    Sby1 = new double[bypassSize];
    Sdbar1 = new double[defaultSize];
    Sbybar1 = new double[bypassSize];

    hmainvec = new double[mainSize];
    hdvec = new double[defaultSize];
    hbyvec = new double[bypassSize];

    interp1(x, r, xmain, xlen, Nmaintotal, Smainorig);
    for (i = 0; i < Nmaintotal; i++) {
	Smainorig[i] = M_PI * Smainorig[i] * Smainorig[i];
    }
    Smainbar[0] = M_PI * r[0] * r[0];
    for (i = 1; i < Nmain[0]; i++) {
	Smainbar[i] = 0.5 * (Smainorig[i-1] + Smainorig[i]);
    }
    Smainbar[Nmain[0]] = Smainorig[Nmain[0] - 1];

    interp1(x, r, xd, xlen, Ndtotal, Sdorig);
    for (i = 0; i < Ndtotal; i++) {
	Sdorig[i] = M_PI * Sdorig[i] * Sdorig[i];
    }

    memset(Sbyti, 0, bypassSize * sizeof(double));

    memset(hmainvec, 0, mainSize * sizeof(double));
    memset(hdvec, 0, defaultSize * sizeof(double));
    memset(hbyvec, 0, bypassSize * sizeof(double));

    energy = NULL;
    if (gs->getEnergyOn()) {
	energy = new double[K];
	pmainvec = new double[mainSize];
	vmainvec = new double[mainSize];
	pdvec = new double[defaultSize];
	vdvec = new double[defaultSize];
	pbyvec = new double[bypassSize];
	vbyvec = new double[bypassSize];
	pdvec1 = new double[defaultSize];
	vdvec1 = new double[defaultSize];
	pbyvec1 = new double[bypassSize];
	vbyvec1 = new double[bypassSize];
    }

    /*
     * don't bother with the 'inv' versions, just compute them on-the-fly
     * when needed
     * and don't do p*vec and v*vec yet, they're only needed for energy calc
     */

    memcpy(Smain, Smainorig, (mjout[0]) * sizeof(double));
    memcpy(Sdti, Sdorig, (djout[0]) * sizeof(double));
    smonext = mjout[0];
    sdonext = djout[0];

    for (i = 0; i < Nmain[0] + 1; i++) {
	hmainvec[i] = hmain[0];
	if (energy) {
	    if ((i == 0) || (i == Nmain[0])) {
		pmainvec[i] = 0.5 * hmain[0];
	    }
	    else {
		pmainvec[i] = hmain[0];
	    }
	}
    }

    for (i = 0; i < Nvalve; i++) {
	/* Smain */
	Smain[mjout[i]] = 1.0;
	memcpy(&Smain[mjout[i]+1], &Smainorig[smonext], ((mainSize - (mjout[i]+1)) * sizeof(double)));
	smonext += (mjout[i+1] - mjout[i]);

	memcpy(Smainorig, Smain, mainSize * sizeof(double));

	/* Sd */
	if (i < (Nvalve - 1)) end = djout[i+1];
	else end = Ndtotal - 1;
	Sdti[djout[i]] = 1.0;
	if ((end - djout[i]) > 0) {
	    memcpy(&Sdti[djout[i]+1], &Sdorig[sdonext], ((defaultSize - (djout[i]+1)) * sizeof(double)));
	    sdonext += (end - djout[i]);
	}

	memcpy(Sdorig, Sdti, defaultSize * sizeof(double));

	/* Sby */
	for (j = byjin[i]; j < byjout[i]; j++) {
	    Sbyti[j] = Smain[mjout[i]-1] + ((Smain[mjin[i+1]] - Smain[mjout[i]-1]) / ((double)(Nby[i]-1))) *
		((double)(j - byjin[i]));
	}
	Sbyti[byjout[i]] = 1.0;

	/* Smainbar */
	Smainbar[mjin[i+1]] = Smain[mjin[i+1]];
	for (j = (mjin[i+1] + 1); j < mjout[i+1]; j++) {
	    Smainbar[j] = 0.5 * (Smain[j-1] + Smain[j]);
	}
	Smainbar[mjout[i+1]] = Smain[mjout[i+1] - 1];

	/* Sdbar */
	Sdbarti[djin[i]] = Sdti[djin[i]];
	for (j = djin[i] + 1; j < djout[i]; j++) {
	    Sdbarti[j] = 0.5 * (Sdti[j-1] + Sdti[j]);
	}
	Sdbarti[djout[i]] = Sdti[djout[i] - 1];

	/* Sbybar */
	Sbybarti[byjin[i]] = Sbyti[byjin[i]];
	for (j = byjin[i] + 1; j < byjout[i]; j++) {
	    Sbybarti[j] = 0.5 * (Sbyti[j-1] + Sbyti[j]);
	}
	Sbybarti[byjout[i]] = Sbyti[byjout[i] - 1];

	/* hmainvec */
	for (j = mjin[i+1]; j <= mjout[i+1]; j++) {
	    hmainvec[j] = hmain[i+1];
	    if (energy) {
		if ((j == mjin[i+1]) || (j == mjout[i+1])) {
		    pmainvec[j] = 0.5 * hmain[i+1];
		}
		else {
		    pmainvec[j] = hmain[i+1];
		}
	    }
	}

	/* hdvec */
	for (j = djin[i]; j <= djout[i]; j++) {
	    hdvec[j] = hd[i];
	    if (energy) {
		if ((j == djin[i]) || (j == djout[i])) {
		    pdvec[j] = 0.5 * hd[i];
		}
		else {
		    pdvec[j] = hd[i];
		}
	    }
	}

	/* hbyvec */
	for (j = byjin[i]; j <= byjout[i]; j++) {
	    hbyvec[j] = hby[i];
	    if (energy) {
		if ((j == byjin[i]) || (j == byjout[i])) {
		    pbyvec[j] = 0.5 * hby[i];
		}
		else {
		    pbyvec[j] = hby[i];
		}
	    }
	}
    }

    Smain[mainSize - 1] = 1.0;
    Smainbar[mainSize - 1] = M_PI * r[xlen-1] * r[xlen-1];

    startupProfiler.end(1);

    /*
     * Visctherm stuff
     */
    startupProfiler.start(2);
    logMessage(1, "Getting filter co-efficients");
    B = new double[historySize];
    A = new double[historySize];
    Adiff = new double[historySize];
    Asum = new double[historySize];
    Bsum = new double[historySize];
    viscthermcoeff(historySize, FS, VTT_TUSTIN, B, A);
    if (loss == 0) {
	B[0] = 0.0;
    }
    for (i = 0; i < (historySize-1); i++) {
	Adiff[i] = A[i] - A[i+1];
	Asum[i] = A[i] + A[i+1];
	Bsum[i] = B[i] + B[i+1];
    }
    Adiff[i] = A[i];
    Asum[i] = A[i];
    Bsum[i] = B[i];
    startupProfiler.end(2);

    
    /*
     * Radiation
     */
    startupProfiler.start(3);
    logMessage(1, "Setting up for radiation");
    RLCconstants(rho, c, sqrt(Smainbar[mainSize - 1] / M_PI), &R1, &R2, &Cr, &Lr);
    R2 = 1.0 / R2;

    if (loss == 0) {
	R1 = 0.0;
	R2 = 0.0;
    }

    ra = pow((1.0 + 2.0 * (rho * c * c * kk / hmain[Nvalve]) * (0.25 * kk / Lr + (0.5 * R2 + Cr / kk) *
								pow((0.5 * (R1*R2 + 1.0) + R1*Cr/kk), -1.0) * 0.5)), -1.0);
    rb = ra * (1.0 - 2.0 * (rho*c*c*kk / hmain[Nvalve]) * (0.25 * kk / Lr + (0.5 * R2 + Cr / kk) *
							   pow((0.5 * (R1*R2 + 1.0) + R1*Cr/kk), -1.0) * 0.5));
    rc = ra * (2.0*rho*c*c*kk*Smain[mainSize-2] / (Smainbar[mainSize-1] * hmain[Nvalve]));
    rd = -ra * 2.0 * rho * c * c * kk / hmain[Nvalve];
    re = -ra * (2.0*rho*c*c*kk / hmain[Nvalve]) * (0.5*R2 - Cr/kk + (0.5*R2 + Cr/kk) *
						   pow((0.5*(R1*R2+1.0) + R1*Cr/kk), -1.0) *
						   (R1*Cr/kk - 0.5 * (R1*R2+1.0)));
    rf = (0.5 * kk / Lr);
    rg = pow((0.5 * (R1*R2+1.0) + R1*Cr/kk), -1.0);
    rh = rg * (R1*Cr/kk - 0.5*(R1*R2+1.0));
    rg = 0.5 * rg;
    startupProfiler.end(3);

    if (energy) {
	bha = 0.5 * Smainbar[mainSize-1] * Lr;
	bhb = 0.5 * ((2.0*R1*R2+1.0)*Cr) * Smainbar[mainSize-1];

	for (i = 0; i < mainSize; i++) {
	    pmainvec[i] = Smainbar[i] * pmainvec[i] / (2.0*rho*c*c);
	    vmainvec[i] = rho * hmainvec[i] * Smain[i] / 2.0;
	}
	for (i = 0; i < (Nvalve+1); i++) {
	    vmainvec[mjout[i]] = 0.0;
	}
    }

    /*
     * Precalculated parts for tube update
     */
    startupProfiler.start(4);
    logMessage(1, "Precalculating for main tube updates");
    /* main pressure update */
    pmaina = new double[mainSize - 2];
    pmainb = new double[mainSize - 2];
    pmainc = new double[mainSize - 2];

    for (i = 0; i < (mainSize - 2); i++) {
	double pmaine;
	pmaina[i] = 1.0 / (A[0] + (rho*c*c*kk)*(gamma-1.0)*sqrt(eta*M_PI / (rho*rho*rho*Smainbar[i+1])) *
			   B[0]/(nu*c*c));
	pmainb[i] = -pmaina[i] * ((rho*c*c*kk)*(gamma-1.0)*sqrt(eta*M_PI / (rho*rho*rho*Smainbar[i+1])) /
				  (nu*c*c));
	pmaine = hmainvec[i+1] * Smainbar[i+1];
	pmainc[i] = pmaina[i] * (rho*c*c*kk) / pmaine;
    }

    /* velocity main tube */
    vmainb = new double[mainSize];
    vmainc = new double[mainSize];
    vmaind = new double[mainSize];
    vmaine = new double[mainSize];

    for (i = 0; i < mainSize; i++) {
	double vmaina;
	vmaina = 1.0 / (A[0] * (rho/kk + 1.5*eta*M_PI/Smain[i]) + B[0]*sqrt(rho*eta*M_PI/Smain[i]));
	vmainb[i] = vmaina * (rho / kk);
	vmainc[i] = -vmaina * (1.5*eta*M_PI / Smain[i]);
	vmaind[i] = -vmaina * sqrt(rho*eta*M_PI / Smain[i]);
	vmaine[i] = vmaina / hmainvec[i];
    }
    vmaine[mainSize-1] = 0.0;
    startupProfiler.end(4);


    /*
     * Allocate arrays
     */
    logMessage(1, "Allocating arrays");
    
    /* compute c1 for the lips */
    double S0 = M_PI * r[0] * r[0];
    c1 = h * S0 / (rho * c * c * kk);

    /* delete temporaries */
    delete[] Lmain;
    delete[] Nmain;
    delete[] Nby;
    delete[] xmain;
    delete[] xd;

    delete[] r;
    delete[] x;

    delete[] Smainorig;
    delete[] Sdorig;
    delete[] hmainvec;

    /* allocate and clear state and history arrays */
    pmain1 = new double[mainSize * (historySize+1)];
    vmain1 = new double[mainSize * (historySize+1)];

    pd1 = new double[defaultSize * (historySize+1)];
    vd1 = new double[defaultSize * (historySize+1)];

    pby1 = new double[bypassSize * (historySize+1)];
    vby1 = new double[bypassSize * (historySize+1)];

    memset(pmain1, 0, mainSize * (historySize+1) * sizeof(double));
    memset(vmain1, 0, mainSize * (historySize+1) * sizeof(double));

    memset(pd1, 0, defaultSize * (historySize+1) * sizeof(double));
    memset(vd1, 0, defaultSize * (historySize+1) * sizeof(double));

    memset(pby1, 0, bypassSize * (historySize+1) * sizeof(double));
    memset(vby1, 0, bypassSize * (historySize+1) * sizeof(double));

    /*
     * Since we do our own state and history allocation, need to handle
     * impulses here as Component class won't do it for us
     */
    if (sm->getBoolSetting(name, "impulse")) {
	// add impulse in centre of main tube
	pmain1[(mainSize * 3) / 2] = 1.0;
    }

    /*
     * Set history and state pointers
     */
    pmain = pmain1;
    vmain = vmain1;
    pd = pd1;
    vd = vd1;
    pby = pby1;
    vby = vby1;

    pmainhist = new double*[historySize];
    vmainhist = new double*[historySize];
    pdhist = new double*[historySize];
    vdhist = new double*[historySize];
    pbyhist = new double*[historySize];
    vbyhist = new double*[historySize];

    for (i = 0; i < historySize; i++) {
	pmainhist[i] = &pmain1[mainSize * (i+1)];
	vmainhist[i] = &vmain1[mainSize * (i+1)];
	pdhist[i] = &pd1[defaultSize * (i+1)];
	vdhist[i] = &vd1[defaultSize * (i+1)];
	pbyhist[i] = &pby1[bypassSize * (i+1)];
	vbyhist[i] = &vby1[bypassSize * (i+1)];
    }

    /*
     * Set the standard component members so that inputs and
     * outputs work
     */
    ss = mainSize;
    u = pmain;
    u1 = pmainhist[0];
    u2 = pmainhist[1];

    /*
     * Space for valve data pointers
     */
    valveQd = new double[Nvalve];
    valves = new InputValve*[Nvalve];

#ifdef USE_AVX
    if (gs->getAVXEnabled()) {
	/*
	 * Allocate aligned and duplicated arrays for AVX
	 */
	A4_orig = new double[(historySize + 1) * 4];
	Adiff4_orig = new double[(historySize + 1) * 4];
	Bsum4_orig = new double[(historySize + 1) * 4];
	Asum4_orig = new double[(historySize + 1) * 4];

	A4 = A4_orig;
	while (((long)A4) & 0x1f) A4++;
	Adiff4 = Adiff4_orig;
	while (((long)Adiff4) & 0x1f) Adiff4++;
	Bsum4 = Bsum4_orig;
	while (((long)Bsum4) & 0x1f) Bsum4++;
	Asum4 = Asum4_orig;
	while (((long)Asum4) & 0x1f) Asum4++;

	for (i = 0; i < historySize; i++) {
	    A4[i*4] = A[i];
	    A4[i*4+1] = A[i];
	    A4[i*4+2] = A[i];
	    A4[i*4+3] = A[i];

	    Adiff4[i*4] = Adiff[i];
	    Adiff4[i*4+1] = Adiff[i];
	    Adiff4[i*4+2] = Adiff[i];
	    Adiff4[i*4+3] = Adiff[i];

	    Bsum4[i*4] = Bsum[i];
	    Bsum4[i*4+1] = Bsum[i];
	    Bsum4[i*4+2] = Bsum[i];
	    Bsum4[i*4+3] = Bsum[i];

	    Asum4[i*4] = Asum[i];
	    Asum4[i*4+1] = Asum[i];
	    Asum4[i*4+2] = Asum[i];
	    Asum4[i*4+3] = Asum[i];
	}

	/* align pmain* */
	pmaina_orig = new double[mainSize+2];
	pmainb_orig = new double[mainSize+2];
	pmainc_orig = new double[mainSize+2];

	double *ptmp = pmaina_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, pmaina, (mainSize-2)*sizeof(double));
	delete[] pmaina;
	pmaina = ptmp;

	ptmp = pmainb_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, pmainb, (mainSize-2)*sizeof(double));
	delete[] pmainb;
	pmainb = ptmp;

	ptmp = pmainc_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, pmainc, (mainSize-2)*sizeof(double));
	delete[] pmainc;
	pmainc = ptmp;

	/* align Smain */
	Smain_orig = new double[mainSize+4];
	ptmp = Smain_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, Smain, mainSize*sizeof(double));
	delete[] Smain;
	Smain = ptmp;

	/* align vmain* */
	vmainb_orig = new double[mainSize+4];
	vmainc_orig = new double[mainSize+4];
	vmaind_orig = new double[mainSize+4];
	vmaine_orig = new double[mainSize+4];

	ptmp = vmainb_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, vmainb, (mainSize)*sizeof(double));
	delete[] vmainb;
	vmainb = ptmp;

	ptmp = vmainc_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, vmainc, (mainSize)*sizeof(double));
	delete[] vmainc;
	vmainc = ptmp;

	ptmp = vmaind_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, vmaind, (mainSize)*sizeof(double));
	delete[] vmaind;
	vmaind = ptmp;

	ptmp = vmaine_orig;
	while (((long)ptmp) & 0x1f) ptmp++;
	memcpy(ptmp, vmaine, (mainSize)*sizeof(double));
	delete[] vmaine;
	vmaine = ptmp;

	/* pre-compute scalars */
	etaPiDivRho3 = new double[4];
	gammaM1DivNuCC = new double[4];
	b0 = new double[4];
	a0DivRhoCCkk = new double[4];
	invRhoCCkk = new double[4];

	rhoDivKk = new double[4];
	oneP5EtaPi = new double[4];
	rhoEtaPi = new double[4];

	etaPiDivRho3[0] = (eta * M_PI) / (rho*rho*rho);
	etaPiDivRho3[1] = (eta * M_PI) / (rho*rho*rho);
	etaPiDivRho3[2] = (eta * M_PI) / (rho*rho*rho);
	etaPiDivRho3[3] = (eta * M_PI) / (rho*rho*rho);

	gammaM1DivNuCC[0] = (gamma - 1.0) / (nu * c * c);
	gammaM1DivNuCC[1] = (gamma - 1.0) / (nu * c * c);
	gammaM1DivNuCC[2] = (gamma - 1.0) / (nu * c * c);
	gammaM1DivNuCC[3] = (gamma - 1.0) / (nu * c * c);

	b0[0] = B[0];
	b0[1] = B[0];
	b0[2] = B[0];
	b0[3] = B[0];

	a0DivRhoCCkk[0] = A[0] / (rho*c*c*kk);
	a0DivRhoCCkk[1] = A[0] / (rho*c*c*kk);
	a0DivRhoCCkk[2] = A[0] / (rho*c*c*kk);
	a0DivRhoCCkk[3] = A[0] / (rho*c*c*kk);

	invRhoCCkk[0] = 1.0 / (rho*c*c*kk);
	invRhoCCkk[1] = 1.0 / (rho*c*c*kk);
	invRhoCCkk[2] = 1.0 / (rho*c*c*kk);
	invRhoCCkk[3] = 1.0 / (rho*c*c*kk);

	rhoDivKk[0] = rho / kk;
	rhoDivKk[1] = rho / kk;
	rhoDivKk[2] = rho / kk;
	rhoDivKk[3] = rho / kk;

	oneP5EtaPi[0] = 1.5 * eta * M_PI;
	oneP5EtaPi[1] = 1.5 * eta * M_PI;
	oneP5EtaPi[2] = 1.5 * eta * M_PI;
	oneP5EtaPi[3] = 1.5 * eta * M_PI;

	rhoEtaPi[0] = rho * eta * M_PI;
	rhoEtaPi[1] = rho * eta * M_PI;
	rhoEtaPi[2] = rho * eta * M_PI;
	rhoEtaPi[3] = rho * eta * M_PI;
    }
#endif
}

BrassInstrument::~BrassInstrument()
{
    delete[] mjin;
    delete[] mjout;
    delete[] djin;
    delete[] djout;
    delete[] byjin;
    delete[] byjout;

    delete[] A;
    delete[] Adiff;
    delete[] Asum;
    delete[] Bsum;

    delete[] pmain1;
    delete[] vmain1;
    delete[] pmainhist;
    delete[] vmainhist;

    delete[] pd1;
    delete[] vd1;
    delete[] pdhist;
    delete[] vdhist;

    delete[] pby1;
    delete[] vby1;
    delete[] pbyhist;
    delete[] vbyhist;

    delete[] tempd1;
    delete[] tempdbar1;
    delete[] tempby1;
    delete[] tempbybar1;

    delete[] Sd1;
    delete[] Sby1;
    delete[] Sdbar1;
    delete[] Sbybar1;

    // so Component destructor doesn't try to free them
    u = NULL;
    u1 = NULL;
    u2 = NULL;

    if (energy) {
	delete[] energy;
	delete[] pmainvec;
	delete[] vmainvec;
	delete[] pdvec;
	delete[] vdvec;
	delete[] pbyvec;
	delete[] vbyvec;
	delete[] pdvec1;
	delete[] vdvec1;
	delete[] pbyvec1;
	delete[] vbyvec1;
    }

#ifdef USE_AVX
    if (GlobalSettings::getInstance()->getAVXEnabled()) {
	delete[] A4_orig;
	delete[] Adiff4_orig;
	delete[] Bsum4_orig;
	delete[] Asum4_orig;

	delete[] pmaina_orig;
	delete[] pmainb_orig;
	delete[] pmainc_orig;

	delete[] Smain_orig;

	delete[] vmainb_orig;
	delete[] vmainc_orig;
	delete[] vmaind_orig;
	delete[] vmaine_orig;

	delete[] etaPiDivRho3;
	delete[] gammaM1DivNuCC;
	delete[] b0;
	delete[] a0DivRhoCCkk;
	delete[] invRhoCCkk;

	delete[] rhoDivKk;
	delete[] oneP5EtaPi;
	delete[] rhoEtaPi;
    }
    else {
#endif
    delete[] pmaina;
    delete[] pmainb;
    delete[] pmainc;

    delete[] Smain;

    delete[] vmainb;
    delete[] vmainc;
    delete[] vmaind;
    delete[] vmaine;
#ifdef USE_AVX
    }
#endif

    delete[] valveQd;

    delete[] Nd;
    delete[] Sdti;
    delete[] Sdbarti;
    delete[] Sbyti;
    delete[] Sbybarti;
    delete[] Smainbar;
    delete[] hdvec;
    delete[] hbyvec;
    delete[] hmain;
    delete[] hd;
    delete[] hby;
    delete[] B;

    //printf("Brass profile: %s\n", profiler.print().c_str());
    //printf("Brass startup profile: %s\n", startupProfiler.print().c_str());
}

void BrassInstrument::RLCconstants(double rho, double c, double a, double *R1, double *R2, double *C, double *L)
{
    *R1 = rho * c;
    *R2 = 0.505 * rho * c;
    *C = 1.111 * a / (rho * c * c);
    *L = 0.613 * rho * a;
}

void BrassInstrument::viscthermcoeff(int M, int FS, int type, double *N, double *D)
{
    int MM = 2 * M - 1;
    double alpha = 0.5;
    double *taylnum = new double[MM];
    double *taylden = new double[MM];
    double fac;
    int i, j;
    double prod;
    double tmp;

    double *a, *b, *dummy;
    double *c, *Ntmp, *Dtmp, *Nnext, *Dnext;

    memset(taylnum, 0, MM * sizeof(double));
    memset(taylden, 0, MM * sizeof(double));
    taylnum[0] = 1.0;
    taylden[0] = 1.0;

    switch (type) {
    case VTT_TUSTIN:
    case VTT_AL:
	fac = sqrt(2.0 * (double)FS);
	if (type == VTT_AL) {
	    fac = sqrt((8.0 * (double)FS) / 7.0);
	}
	for (i = 1; i < MM; i++) {
	    prod = 1.0;
	    tmp = alpha - ((double)i) + 1.0;
	    while (tmp <= alpha) {
		prod *= tmp;
		tmp += 1.0;
	    }

	    if (type == VTT_TUSTIN) {
		taylnum[i] = (pow(-1.0, (double)i) * prod) / factorial((double)i);
		taylden[i] = prod / factorial((double)i);
	    }
	    else { /* VTT_AL */
		taylnum[i] = (pow(-1.0, (double)i) * prod) / factorial((double)i);
		taylden[i] = (pow(1.0/7.0, (double)i) * prod) / factorial((double)i);
	    }
	}

	a = taylnum;
	b = taylden;
	c = new double[MM];
	dummy = new double[MM];
	for (i = 0; i < MM; i++) {
	    c[i] = a[0] / b[0];
	    for (j = 0; j < (MM-1); j++) {
		dummy[j] = (-b[j+1] * c[i]) + a[j+1];
	    }
	    dummy[j] = 0.0;
	    memcpy(a, b, MM * sizeof(double));
	    memcpy(b, dummy, MM * sizeof(double));
	}
	
	Ntmp = new double[MM];
	Dtmp = new double[MM];
	memset(Ntmp, 0, MM * sizeof(double));
	memset(Dtmp, 0, MM * sizeof(double));

	Nnext = new double[MM];
	Dnext = new double[MM];
	Ntmp[0] = c[MM-1];
	Dtmp[0] = 1.0;

	for (i = 0; i < (MM-1); i++) {
	    memcpy(Dnext, Ntmp, MM * sizeof(double));
	    Nnext[0] = c[(MM-2)-i] * Ntmp[0];
	    for (j = 1; j < MM; j++) {
		Nnext[j] = c[(MM-2)-i] * Ntmp[j] + Dtmp[j-1];
	    }
	    memcpy(Dtmp, Dnext, MM * sizeof(double));
	    memcpy(Ntmp, Nnext, MM * sizeof(double));
	}

	for (i = 0; i < M; i++) {
	    D[i] = Dtmp[i] / Ntmp[0];
	    N[i] = (fac * Ntmp[i]) / Ntmp[0];
	}
	delete[] taylnum;
	delete[] taylden;
	delete[] c;
	delete[] dummy;
	delete[] Ntmp;
	delete[] Dtmp;
	delete[] Nnext;
	delete[] Dnext;
	break;
    case VTT_GL:
	memset(N, 0, M * sizeof(double));
	memset(D, 0, M * sizeof(double));
	N[0] = 1.0;
	D[0] = 1.0;
	for (i = 1; i < M; i++) {
	    N[i] = (1.0 - (1.5 / (double)i)) * N[i-1];
	}
	for (i = 0; i < M; i++) {
	    N[i] = N[i] * sqrt((double)FS);
	}
	break;
    default:
	logMessage(5, "Invalid visctherm type for brass instrument");
	exit(1);
    }

}


void BrassInstrument::setValveData(int valve, InputValve *iv)
{
    valves[valve] = iv;

    valvesSet++;
    if (valvesSet > Nvalve) {
	logMessage(5, "Error: too many valve inputs set for instrument %s!\n", name.c_str());
	exit(1);
    }
}


#define PMAIN_HIST(ts, loc) (pmainhist[ts][loc])
#define VMAIN_HIST(ts, loc) (vmainhist[ts][loc])

#define PD_HIST(ts, loc) (pdhist[ts][loc])
#define VD_HIST(ts, loc) (vdhist[ts][loc])

#define PBY_HIST(ts, loc) (pbyhist[ts][loc])
#define VBY_HIST(ts, loc) (vbyhist[ts][loc])



void BrassInstrument::runTimestep(int n)
{
    int i = n, j, k;
    double val;
    GlobalSettings *gs = GlobalSettings::getInstance();
    int K = gs->getNumTimesteps();
    bool avx = gs->getAVXEnabled();
    double kk = GlobalSettings::getInstance()->getK();


    /* get valve positions for this timestep */
    for (i = 0; i < Nvalve; i++) {
	valveQd[i] = valves[i]->nextQd();
    }

    /* compute temp* for this timestep */
    for (i = 0; i < Nvalve; i++) {
	for (j = djin[i]; j <= djout[i]; j++) {
	    tempd1[j] = valveQd[i];
	}
	tempdbar1[djin[i]] = valveQd[i];
	tempdbar1[djout[i]] = valveQd[i];
	for (j = (djin[i]+1); j < djout[i]; j++) {
	    tempdbar1[j] = 0.5 * (tempd1[j-1] + tempd1[j]);
	}

	for (j = byjin[i]; j <= byjout[i]; j++) {
	    tempby1[j] = 1.0;
	}
	for (j = 0; j < Nd[i]; j++) {
	    tempby1[byjin[i]+j] = (1.0 - valveQd[i]);
	    tempby1[byjout[i]-j] = (1.0 - valveQd[i]);
	}
	tempby1[byjout[i]-j] = (1.0 - valveQd[i]);
	tempbybar1[byjin[i]] = (1.0 - valveQd[i]);
	tempbybar1[byjout[i]] = (1.0 - valveQd[i]);
	for (j = (byjin[i]+1); j < byjout[i]; j++) {
	    tempbybar1[j] = 0.5 * (tempby1[j-1] + tempby1[j]);
	}
    }

    /* compute S* for this timestep */
    for (i = 0; i < defaultSize; i++) {
	Sd1[i] = tempd1[i] * Sdti[i];
	Sdbar1[i] = tempdbar1[i] * Sdbarti[i];
	tempd1[i] = ceil(tempd1[i]);
	tempdbar1[i] = ceil(tempdbar1[i]);
	if (tempd1[i] == 0.0) Sd1[i] = 1.0;
	if (tempdbar1[i] == 0.0) Sdbar1[i] = 1.0;
    }
    for (i = 0; i < bypassSize; i++) {
 	Sby1[i] = tempby1[i] * Sbyti[i];
	Sbybar1[i] = tempbybar1[i] * Sbybarti[i];
	tempby1[i] = ceil(tempby1[i]);
	tempbybar1[i] = ceil(tempbybar1[i]);
	if (tempby1[i] == 0.0) Sby1[i] = 1.0;
	if (tempbybar1[i] == 0.0) Sbybar1[i] = 1.0;
    }

    /* runInputs parameters have different meanings for the brass inputs! */
    i = n;
    profiler.start(0);
    runInputs(n, pmain, &PMAIN_HIST(0, 0), &VMAIN_HIST(0, 0));
    profiler.end(0);

    /*
     * Tube pressure
     */
    /* main */
    profiler.start(1);
#ifdef USE_AVX
    if (avx) {
	mainPressureUpdateAVX();
    }
    else {
#endif
    for (j = 1; j < (mainSize - 1); j++) {
	val = 0.0;

	for (k = 0; k < historySize; k++) {
	    val += pmaina[j-1] * (Adiff[k] * PMAIN_HIST(k, j));
	    val += pmainb[j-1] * (Bsum[k]  * PMAIN_HIST(k, j));
	    val += pmainc[j-1] * Smain[j-1] * (A[k] * VMAIN_HIST(k, j-1));
	    val -= pmainc[j-1] * Smain[j] * (A[k] * VMAIN_HIST(k, j));
	}

	pmain[j] = val;
    }
#ifdef USE_AVX
    }
#endif
    profiler.end(1);
	
    /* default tube */
    profiler.start(4);
    for (j = 1; j < (defaultSize - 1); j++) {
	double da, db, dc, de, de1, de2;

	da = tempdbar1[j] /
	    (A[0]/(rho*c*c*kk) + (gamma-1.0)*sqrt(eta*M_PI/(rho*rho*rho*Sdbar1[j]))*B[0] / (nu*c*c));
	db = da / (rho*c*c*kk);
	dc = -da * ((gamma-1.0)*sqrt(eta*M_PI/(rho*rho*rho*Sdbar1[j])) / (nu*c*c));
	de = hdvec[j] * Sdbar1[j];
	de1 = da * tempd1[j-1] * Sd1[j-1] / de;
	de2 = -da * tempd1[j] * Sd1[j] / de;

	val = 0.0;

	for (k = 0; k < historySize; k++) {
	    val += db * (Adiff[k] * PD_HIST(k, j));
	    val += dc * (Bsum[k]  * PD_HIST(k, j));
	    val += de1 * (A[k] * VD_HIST(k, j-1));
	    val += de2 * (A[k] * VD_HIST(k, j));
	}

	pd[j] = val;
    }
    profiler.end(4);

    /* bypass tube pressure */
    profiler.start(5);
#ifdef USE_AVX
    if (avx) {
	bypassPressureUpdateAVX(i);
    }
    else {
#endif
    for (j = 1; j < (bypassSize - 1); j++) {
	double bya, byb, byc, byd, byd1, byd2;

	bya = tempbybar1[j] /
	    (A[0]/(rho*c*c*kk) + (gamma-1.0)*sqrt(eta*M_PI/(rho*rho*rho*Sbybar1[j]))*B[0] / (nu*c*c));
	byb = bya / (rho*c*c*kk);
	byc = -bya * ((gamma-1.0)*sqrt(eta*M_PI/(rho*rho*rho*Sbybar1[j])) / (nu*c*c));
	byd = hbyvec[j] * Sbybar1[j];
	byd1 = bya * tempby1[j-1] * Sby1[j-1] / byd;
	byd2 = -bya * tempby1[j] * Sby1[j] / byd;

	val = 0.0;

	for (k = 0; k < historySize; k++) {
	    val += byb * (Adiff[k] * PBY_HIST(k, j));
	    val += byc * (Bsum[k]  * PBY_HIST(k, j));
	    val += byd1 * (A[k] * VBY_HIST(k, j-1));
	    val += byd2 * (A[k] * VBY_HIST(k, j));
	}

	pby[j] = val;
    }
#ifdef USE_AVX
    }
#endif
    profiler.end(5);
	
    profiler.start(2);
    for (j = 1; j < (Nvalve+1); j++) {
	/* loop over mjin entries */
	val = 0.0;
	
	double qd = valveQd[j-1];
	double mjina = 1.0 / ((hmain[j] + hd[j-1]*qd + hby[j-1]*(1.0-qd)) *
			      Smainbar[mjin[j]]*A[0]/(rho*c*c*kk) +
			      (hmain[j] + hd[j-1]*sqrt(qd) + hby[j-1]*sqrt((1.0-qd))) *
			      (B[0]*(gamma-1.0)*sqrt(eta*M_PI*Smainbar[mjin[j]] / (rho*rho*rho)) /
			       (nu*c*c)));
	double mjinb = mjina * ((hmain[j] + hd[j-1]*qd + hby[j-1]*(1.0-qd)) *
				(Smainbar[mjin[j]]) / (rho*c*c*kk));
	double mjinc = -mjina * ((hmain[j] + hd[j-1]*sqrt(qd) + hby[j-1]*sqrt((1.0-qd))) *
				 (gamma-1.0) * sqrt(eta*M_PI*Smainbar[mjin[j]] / (rho*rho*rho)) /
				 (nu*c*c));

	for (k = 0; k < historySize; k++) {
	    val += mjinb * (Adiff[k] * PMAIN_HIST(k, mjin[j]));
	    val += mjinc * (Bsum[k]  * PMAIN_HIST(k, mjin[j]));
	    val += 2.0 * mjina *
		((tempd1[(djout[j-1]-1)] * Sd1[(djout[j-1]-1)] *
		  A[k] * VD_HIST(k, djout[j-1]-1)) +
		 (tempby1[(byjout[j-1]-1)] * Sby1[(byjout[j-1]-1)] *
		  A[k] * VBY_HIST(k, byjout[j-1]-1)));
	    val -= 2.0 * mjina * Smain[mjin[j]] * A[k] * VMAIN_HIST(k, mjin[j]);
	}

	pmain[mjin[j]] = val;
    }
    profiler.end(2);

    /* outputs at junctions (same as input of default and bypass) and radiation condition */
    profiler.start(3);
    for (j = 0; j < Nvalve; j++) {
	/* loop over mjout entries */
	val = 0.0;

	double qd = valveQd[j];
	double mjouta = 1.0 / ((hmain[j] + hd[j]*qd + hby[j]*(1.0-qd)) *
			       Smainbar[mjout[j]]*A[0]/(rho*c*c*kk) +
			       (hmain[j] + hd[j]*sqrt(qd) + hby[j]*sqrt((1.0-qd))) *
			       (B[0]*(gamma-1.0)*sqrt(eta*M_PI*Smainbar[mjout[j]] / (rho*rho*rho)) /
				(nu*c*c)));
	double mjoutb = mjouta * ((hmain[j] + hd[j]*qd + hby[j]*(1.0-qd)) *
				  (Smainbar[mjout[j]]) / (rho*c*c*kk));
	double mjoutc = -mjouta * ((hmain[j] + hd[j]*sqrt(qd) + hby[j]*sqrt((1.0-qd))) *
				   (gamma-1.0) * sqrt(eta*M_PI*Smainbar[mjout[j]] / (rho*rho*rho)) /
				   (nu*c*c));

	for (k = 0; k < historySize; k++) {
	    val += mjoutb * (Adiff[k] * PMAIN_HIST(k, mjout[j]));
	    val += mjoutc * (Bsum[k]  * PMAIN_HIST(k, mjout[j]));
	    val += 2.0 * mjouta * Smain[mjout[j]-1] * A[k] * VMAIN_HIST(k, mjout[j]-1);
	    val -= 2.0 * mjouta * tempd1[djin[j]] * Sd1[djin[j]] *
		A[k] * VD_HIST(k, djin[j]);
	    val -= 2.0 * mjouta * tempby1[byjin[j]] * Sby1[byjin[j]] *
		A[k] * VBY_HIST(k, byjin[j]);
	}

	pmain[mjout[j]] = val;
    }
    pmain[mjout[Nvalve]] = rb * PMAIN_HIST(0, mainSize-1) + rc * VMAIN_HIST(0, mainSize-2) + rd*vr + re*pr;
    profiler.end(3);
	
    /* bypass junctions output and input */
    for (j = 0; j < Nvalve; j++) {
	pby[byjin[j]] = tempbybar1[byjin[j]] * pmain[mjout[j]];
    }
    for (j = 0; j < Nvalve; j++) {
	pby[byjout[j]] = tempbybar1[byjout[j]] * pmain[mjin[j+1]];
    }

    /* default tube junctions output and input */
    for (j = 0; j < Nvalve; j++) {
	pd[djin[j]] = tempdbar1[djin[j]] * pmain[mjout[j]];
    }
    for (j = 0; j < Nvalve; j++) {
	pd[djout[j]] = tempdbar1[djout[j]] * pmain[mjin[j+1]];
    }

    /* radiation stored variable update */
    vr = vr + rf * (pmain[mainSize-1] + PMAIN_HIST(0, mainSize-1));
    pr = (rh * pr) + rg * (pmain[mainSize-1] + PMAIN_HIST(0, mainSize-1));

    /*
     * Velocity update
     */
    /* main tube */
    profiler.start(6);
#ifdef USE_AVX
    if (avx) {
	mainVelocityUpdateAVX();
    }
    else {
#endif
    for (j = 0; j < (mainSize-1); j++) {

	val = vmainb[j] * (Adiff[0] * VMAIN_HIST(0, j));
	val += vmainc[j] * (Asum[0]  * VMAIN_HIST(0, j));
	val += vmaind[j] * (Bsum[0]  * VMAIN_HIST(0, j));
	val += vmaine[j] * A[0] * (pmain[j] - pmain[j+1]);

	for (k = 1; k < historySize; k++) {
	    val += vmainb[j] * (Adiff[k] * VMAIN_HIST(k, j));
	    val += vmainc[j] * (Asum[k]  * VMAIN_HIST(k, j));
	    val += vmaind[j] * (Bsum[k]  * VMAIN_HIST(k, j));
	    val += vmaine[j] * A[k] * (PMAIN_HIST(k-1, j) - PMAIN_HIST(k-1, j+1));
	}

	vmain[j] = val;
    }
    /* final entry is different. j is still used here, it's now mainSize-1 */
    val = 0.0;
    for (k = 0; k < historySize; k++) {
	val += vmainb[j] * (Adiff[k] * VMAIN_HIST(k, j));
	val += vmainc[j] * (Asum[k]  * VMAIN_HIST(k, j));
	val += vmaind[j] * (Bsum[k]  * VMAIN_HIST(k, j));
    }
    vmain[j] = val;
#ifdef USE_AVX
    }
#endif
    profiler.end(6);


    /* default tube */
    profiler.start(7);
    for (j = 0; j < (defaultSize-1); j++) {
	double da, db, dc, dd, de;
	
	da = tempd1[j] /
	    (A[0] * (rho/kk + 1.5*eta*M_PI/Sd1[j]) + B[0]*sqrt(rho*eta*M_PI/Sd1[j]));
	db = da * (rho / kk);
	dc = -da * (1.5*eta*M_PI / Sd1[j]);
	dd = -da * sqrt(rho*eta*M_PI / Sd1[j]);
	de = da / hdvec[j];
	
	val = db * (Adiff[0] * VD_HIST(0, j));
	val += dc * (Asum[0]  * VD_HIST(0, j));
	val += dd * (Bsum[0]  * VD_HIST(0, j));
	val += de * A[0] * (pd[j] - pd[j+1]);

	for (k = 1; k < historySize; k++) {
	    val += db * (Adiff[k] * VD_HIST(k, j));
	    val += dc * (Asum[k]  * VD_HIST(k, j));
	    val += dd * (Bsum[k]  * VD_HIST(k, j));
	    val += de * A[k] * (PD_HIST(k-1, j) - PD_HIST(k-1, j+1));
	}

	vd[j] = val;
    }
    {
	double da, db, dc, dd;
	da = tempd1[j] /
	    (A[0] * (rho/kk + 1.5*eta*M_PI/Sd1[j]) + B[0]*sqrt(rho*eta*M_PI/Sd1[j]));
	db = da * (rho / kk);
	dc = -da * (1.5*eta*M_PI / Sd1[j]);
	dd = -da * sqrt(rho*eta*M_PI / Sd1[j]);

	val = 0.0;
	for (k = 0; k < historySize; k++) {
	    val += db * (Adiff[k] * VD_HIST(k, j));
	    val += dc * (Asum[k]  * VD_HIST(k, j));
	    val += dd * (Bsum[k]  * VD_HIST(k, j));
	}
	vd[j] = val;
    }
    profiler.end(7);
    

    /* bypass tube */
    profiler.start(8);
#ifdef USE_AVX
    if (avx) {
	bypassVelocityUpdateAVX(i);
    }
    else {
#endif
    for (j = 0; j < (bypassSize-1); j++) {
	double bya, byb, byc, byd, bye;
    
	bya = tempby1[j] /
	    (A[0] * (rho/kk + 1.5*eta*M_PI/Sby1[j]) + B[0]*sqrt(rho*eta*M_PI/Sby1[j]));
	byb = bya * (rho / kk);
	byc = -bya * (1.5*eta*M_PI / Sby1[j]);
	byd = -bya * sqrt(rho*eta*M_PI / Sby1[j]);
	bye = bya / hbyvec[j];

	val = byb * (Adiff[0] * VBY_HIST(0, j));
	val += byc * (Asum[0]  * VBY_HIST(0, j));
	val += byd * (Bsum[0]  * VBY_HIST(0, j));
	val += bye * A[0] * (pby[j] - pby[j+1]);

	for (k = 1; k < historySize; k++) {
	    val += byb * (Adiff[k] * VBY_HIST(k, j));
	    val += byc * (Asum[k]  * VBY_HIST(k, j));
	    val += byd * (Bsum[k]  * VBY_HIST(k, j));
	    val += bye * A[k] * (PBY_HIST(k-1, j) - PBY_HIST(k-1, j+1));
	}

	vby[j] = val;
    }
    {
	double bya, byb, byc, byd;
	bya = tempby1[j] /
	    (A[0] * (rho/kk + 1.5*eta*M_PI/Sby1[j]) + B[0]*sqrt(rho*eta*M_PI/Sby1[j]));
	byb = bya * (rho / kk);
	byc = -bya * (1.5*eta*M_PI / Sby1[j]);
	byd = -bya * sqrt(rho*eta*M_PI / Sby1[j]);

	val = 0.0;
	for (k = 0; k < historySize; k++) {
	    val += byb * (Adiff[k] * VBY_HIST(k, j));
	    val += byc * (Asum[k]  * VBY_HIST(k, j));
	    val += byd * (Bsum[k]  * VBY_HIST(k, j));
	}
	vby[j] = val;
    }
#ifdef USE_AVX
    }
#endif
    profiler.end(8);


    /* compute energy for this timestep */
    if (energy) {
	/* unlike in the original, these arrays are not time-varying */
	for (j = 0; j < defaultSize; j++) {
	    pdvec1[j] = Sdbar1[j] * pdvec[j] / (2.0*rho*c*c);
	    vdvec1[j] = rho * Sd1[j] * hdvec[j] / 2.0;
	    if (tempdbar1[j] == 0.0) pdvec1[j] = 0.0;
	    if (tempd1[j] == 0.0) vdvec1[j] = 0.0;
	}
	for (j = 0; j < bypassSize; j++) {
	    pbyvec1[j] = Sbybar1[j] * pbyvec[j] / (2.0*rho*c*c);
	    vbyvec1[j] = rho * Sby1[j] * hbyvec[j] / 2.0;
	    if (tempbybar1[j] == 0.0) pbyvec1[j] = 0.0;
	    if (tempby1[j] == 0.0) vbyvec1[j] = 0.0;
	}
	for (j = 0; j < Nvalve; j++) {
	    vdvec1[djout[j]] = 0.0;
	    vbyvec1[byjout[j]] = 0.0;
	}

	double etot = 0.0;
	for (j = 0; j < mainSize; j++) {
	    etot += pmainvec[j] * pmain[j] * pmain[j];
	    etot += vmainvec[j] * vmain[j] * vmainhist[0][j];
	}
	for (j = 0; j < defaultSize; j++) {
	    etot += pdvec1[j] * pd[j] * pd[j];
	    etot += vdvec1[j] * vd[j] * vdhist[0][j];
	}
	for (j = 0; j < bypassSize; j++) {
	    etot += pbyvec1[j] * pby[j] * pby[j];
	    etot += vbyvec1[j] * vby[j] * vbyhist[0][j];
	}
	etot += (bha * vr * vr) + (bhb * pr * pr);

	energy[n] = etot;
    }
}

void BrassInstrument::swapBuffers(int n)
{
    double *tpmain, *tvmain, *tpd, *tvd, *tpby, *tvby;
    int i;

    tpmain = pmainhist[historySize - 1];
    tvmain = vmainhist[historySize - 1];
    tpd = pdhist[historySize - 1];
    tvd = vdhist[historySize - 1];
    tpby = pbyhist[historySize - 1];
    tvby = vbyhist[historySize - 1];

    for (i = (historySize - 1); i >= 1; i--) {
	pmainhist[i] = pmainhist[i-1];
	vmainhist[i] = vmainhist[i-1];
	pdhist[i] = pdhist[i-1];
	vdhist[i] = vdhist[i-1];
	pbyhist[i] = pbyhist[i-1];
	vbyhist[i] = vbyhist[i-1];
    }
    pmainhist[0] = pmain;
    vmainhist[0] = vmain;
    pdhist[0] = pd;
    vdhist[0] = vd;
    pbyhist[0] = pby;
    vbyhist[0] = vby;

    pmain = tpmain;
    vmain = tvmain;
    pd = tpd;
    vd = tvd;
    pby = tpby;
    vby = tvby;

    // keep the "standard" pointers updated so that outputs work
    u = pmain;
    u1 = pmainhist[0];
    u2 = pmainhist[1];
}

double BrassInstrument::getC1()
{
    return c1;
}


double *BrassInstrument::getEnergy()
{
    return energy;
}
