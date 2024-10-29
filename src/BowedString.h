/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Bowed string component
 */

#ifndef _BOWEDSTRING_H_
#define _BOWEDSTRING_H_

#include "Component1D.h"
#include "BreakpointFunction.h"
#include "Output.h"

extern "C" {
#include "csrmatrix.h"
};

#include <vector>
#include <cstdio>
using namespace std;

struct Bow {
    BreakpointFunction *f_ext_w;
    BreakpointFunction *f_ext_u;
    BreakpointFunction *pos;
    double yu, yu1, yu2;
    double yw, yw1, yw2;

    // update values for yu
    double By, Cy, Dy;

    // spread and interpolation info
    int Ji, Ji1, Ji2;
    double J[4], J1[4], J2[4];

    double delta, delta1, delta2;

    double P, P1, P2;

    double lastpos;
    double bu;
    double Vrel;

    double Fc, Ff, Psi;
    double JJpJJ2;

    bool isFinger;
    double *vibrato;

    double Kw, Ku, alpha, beta, lambda, M;

    double alp, alm, Ka1, Ka, Kb;

    //Output *Vrel_out;
};

struct InterceptEntry {
    double value;   // value of this entry
    int index;      // index in the original array
    int pad;
};

class BowedString : public Component1D {
 public:
    // constructor for "manual" definitions
    BowedString(string name, double f0, double rho, double rad, double E, double T60_0, double T60_1000, double L);

    // constructor for defined instrument definitions
    // type should be 'violin', 'viola', or 'cello'
    // instrumentIndex can be 0-4 for violin, 0-1 for viola, 0-2 for cello
    // stringIndex should be 0-3
    BowedString(string name, string type, int instrumentIndex, int stringIndex);

    virtual ~BowedString();

    virtual void runTimestep(int n);

    virtual void swapBuffers(int n);

    virtual void logMatrices();
    
    void addBow(double w0, double vw0, double u0, double vu0,
		vector<double> *times, vector<double> *positions,
		vector<double> *forces_w, vector<double> *forces_u,
		vector<double> *vibrato = NULL, bool isFinger = false);

    void addFinger(double w0, double vw0, double u0, double vu0,
		   vector<double> *times, vector<double> *positions,
		   vector<double> *forces_w, vector<double> *forces_u,
		   vector<double> *vibrato);

    virtual double *getEnergy();

    void setBowParameters(double K, double alpha, double beta, double lambda,
			  double M);
    void setFingerParameters(double Kw, double Ku, double alpha, double beta,
			     double lambda, double M);

 protected:
    void init(double f0, double rho, double rad, double E, double T60_0, double T60_1000, double L);

    int getBowInterpInfo(Bow &bow, double coeffs[4], int n = 0);
    void friedlander(double &fric, double Fc, double MF2, Bow &bow);
    void Newton_friction(double &Vrel, double &fric, double F, double b, double MF,
			 double sgn);
    double LagrangeInterp(int xint, double xfrac);
    void coulombNewton(double &fric, double Fc, double MF2, Bow &bow);
    void coulombNewton(double *fric, double *Fc, double *MF2, double *Vrel,
		       double *bu);
    void frictionFingersNeck(double &fric, double &dfric, double velocity);
    double interpolantProduct(int idx1, int idx2, double *J1, double *J2);

    // basic string properties
    double f0;     // fundamental frequency (Hz)
    double T;      // tension (N)
    double L;      // length (m)
    double rho;    // linear mass (kg/m)
    double rad;    // radius (m)
    double E;      // Young's modulus (Pa)

    double I0, lambda1, lambda2;

    // minimum and maximum distance from string to neck
    //double act_min, act_max;

    // backboard parameters for Newton
    //double KN, alphaN, betaN;

    // finger parameters for Newton
    //double KF, alphaF, betaF, MF;

    // bow parameters for Newton
    double KB, alphaB, betaB, MB, lambdaB;

    // finger parameters
    double KwF, KuF, alphaF, betaF, lambdaF, MF;

    double epsr;
    double tol;

    double A;

    // main update matrices
    CSRmatrix *B, *C;

    // additional state arrays for horizontal displacement
    double *w, *w1, *w2;

    // matrices needed for energy calculation
    CSRmatrix *Dxm, *Dxx;

    // timestep length and grid spacing
    double k, h;
    double inv2k;

    // friction table
    static int tablesize;
    static double *fric_table, *dfric_table;
    static double *intercept;
    static double fricmax, interceptmin;

    static InterceptEntry *interceptSorted;

    vector<Bow> bows;

    int numFingers;

    // energy check stuff
    double *energy;
    double *Hws, *Qws, *Hu, *Qus, *Ew, *Eu;
    double *Hwc, *Hw, *Pw, *Qwc, *powsum_w;
    double *Pu, *Quf, *powsum_u;
    double *etmp1, *etmp2;
    double compw, compu;

    double *Hus, *HuB, *QuB;
};

#endif
