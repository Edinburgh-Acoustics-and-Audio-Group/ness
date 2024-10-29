/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Component class for a brass instrument, based on Reg's Matlab code
 */

#ifndef _BRASSINSTRUMENT_H_
#define _BRASSINSTRUMENT_H_

#include "Component1D.h"
#include "Profiler.h"

#include <vector>
using namespace std;


class InputValve;

class BrassInstrument : public Component1D {
 public:
    BrassInstrument(string name, double temperature, int Nvalve, vector<double> &vpos,
		    vector<double> &vdl, vector<double> &vbl, vector<double> &bore);
    BrassInstrument(string name, double temperature, int Nvalve, vector<double> &vpos,
		    vector<double> &vdl, vector<double> &vbl, double xm, vector<double> &x0,
		    double L, double rm, vector<double> &r0, double rb, double fp);
    virtual ~BrassInstrument();

    virtual void runTimestep(int n);

    /* brass stores more timesteps of history so has a custom buffer swap */
    virtual void swapBuffers(int n);

    void setValveData(int valve, InputValve *iv);
    double getC1();
    double getRho() { return rho; }
    double getSmain0() { return Smain[0]; }

    int getNumValves() { return Nvalve; }
    
    double *getVmainHist() {
	return &vmainhist[0][0];
    }

    virtual double *getEnergy();

 protected:
    void init(double temperature, int Nvalve, vector<double> &vpos, vector<double> &vdl,
	      vector<double> &vbl, vector<double> *bore);
    vector<double> *customBore(double xm, vector<double> &x0, double L, double rm,
			       vector<double> &r0, double rb, double fp);

    void viscthermcoeff(int M, int FS, int type, double *N, double *D);
    void RLCconstants(double rho, double c, double a, double *R1, double *R2, double *C, double *L);

    int Nvalve;

    int valvesSet;

    int historySize; /* M in Matlab code */

    int mainSize;
    int defaultSize;
    int bypassSize;

    InputValve **valves;

    /* scalars */
    double vr, pr;
    double ra, rb, rc, rd, re, rf, rg, rh;
    double c1;

    /* constant history-sized arrays */
    double *Adiff, *Bsum, *A, *Asum;

    /* other constant arrays */
    double *Smain, *pmaina, *pmainb, *pmainc;
    double *vmainb, *vmainc, *vmaind, *vmaine;

    /* main state arrays */
    double *pmain, *vmain;
    double *pd, *vd;
    double *pby, *vby;

    /* history arrays */
    double *pmain1, *vmain1;
    double *pd1, *vd1;
    double *pby1, *vby1;

    /* history pointer arrays */
    double **pmainhist, **vmainhist;
    double **pdhist, **vdhist;
    double **pbyhist, **vbyhist;

    /* index sets */
    int *mjin, *mjout, *djin, *djout, *byjin, *byjout;    

    /* temporary values only needed until valve setup is complete */
    int *Nd;
    double *Smainbar, *Sdbarti, *Sbybarti;
    double *Sdti, *Sbyti;
    double *hmain, *hd, *hby;
    double *B;
    double *hdvec, *hbyvec;
    double deltaT, rho, c, eta, nu, gamma;

    Profiler profiler;
    Profiler startupProfiler;

    double *valveQd;

    /* single timestep versions of arrays that were previously precomputed */
    double *tempd1, *tempdbar1, *tempby1, *tempbybar1;
    double *Sd1, *Sby1, *Sdbar1, *Sbybar1;

    /* energy check stuff */
    double *pmainvec, *vmainvec, *pdvec, *vdvec, *pbyvec, *vbyvec;
    double *pdvec1, *vdvec1, *pbyvec1, *vbyvec1;
    double bha, bhb;
    double *energy;

#ifdef USE_AVX
    void mainPressureUpdateAVX();
    void mainVelocityUpdateAVX();
    void bypassPressureUpdateAVX(int i);
    void bypassVelocityUpdateAVX(int i);

    /* helper arrays for AVX - contain 4 copies of each value and are 32-byte aligned */
    double *A4_orig, *Adiff4_orig, *Bsum4_orig, *Asum4_orig;
    double *A4, *Adiff4, *Bsum4, *Asum4;

    /*
     * Original pointers before we re-align them, so that they can be freed
     */
    double *pmaina_orig, *pmainb_orig, *pmainc_orig;
    double *Smain_orig;
    double *vmainb_orig, *vmainc_orig, *vmaind_orig, *vmaine_orig;

    /* pre-computed scalars for AVX (4 copies) */
    double *etaPiDivRho3, *gammaM1DivNuCC, *b0, *a0DivRhoCCkk, *invRhoCCkk;
    double *rhoDivKk, *oneP5EtaPi, *rhoEtaPi;
#endif
};

#endif
