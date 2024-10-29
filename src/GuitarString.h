/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Guitar string component
 */
#ifndef _GUITARSTRING_H_
#define _GUITARSTRING_H_

#include "Component1D.h"
#include "BreakpointFunction.h"
#include "Profiler.h"

extern "C" {
#include "csrmatrix.h"
#include "pcg.h"
};

#include <vector>
using namespace std;

class GuitarString : public Component1D {
 public:
    GuitarString(string name, double L, double E, double T, double r, double rho, double T60_0, double T60_1000);
    virtual ~GuitarString();

    virtual void runTimestep(int n);
    virtual void swapBuffers(int n);

    virtual void logMatrices();

    void setBackboard(double b0, double b1, double b2);
    void setFrets(vector<double> &fretpos, vector<double> &fretheight);
    void setBarrierParams(double K, double alpha, double beta, int itnum = 20, double tol = 1e-12);

    void setFingerParams(double Mf, double Kf, double alphaf, double betaf);
    void addFinger(double uf0, double vf0, vector<double> *times, vector<double> *position,
		   vector<double> *force);

    virtual double *getEnergy();

 protected:
    bool jacobiSolve1(double *r, double *g, CSRmatrix *M, double *qc, double jtol, int jmaxit);
    bool jacobiSolve2(double *r, double *b, CSRmatrix *IMQ, double jtol, int jmaxit);
    void newtonSolve(double *r, double *eta2, double *g, CSRmatrix *M, double *qc, double *Kc,
		     double *alphac, double *coeffc, double *betac, int Nc, int itnum, double tol,
		     double *phi_ra, double *R, int nnn, int qnum);
    void setupCollisions();
    void mapMStructure();

    double L;
    double E;
    double T;
    double radius;
    double rho;
    double T60_0;
    double T60_1000;

    double k;
    double h;
    double sig0;

    CSRmatrix *B, *C;

    Profiler *prof;

    // collision related things
    int bbss;
    double b0, b1, b2;
    int fretnum;
    double *fretpos, *fretheight;
    double Kb, alphab, betab;
    int itnum;
    double tol;

    bool collisionsOn;
    CSRmatrix *Ic, *Jc, *M;

    int Nc; // Newton size
    double *betac, *Kc, *alphac, *coeffc;
    double *eta1, *eta2, *g, *b, *r, *phi_ra, *R;
    double *qc;

    bool initialised;

    // temporaries for collision calculations
    double *utmp;
    double *tmpvec;
    double *Dinv;

    double *maxa, *phi_a, *fac2, *fac3;
    double *phi_prime;
    CSRmatrix *IMQ, *W;

    double *maxrepsinv, *absreps, *notabsreps, *phidiff;
    double *F, *temp, *D;
    int *Dind;
    int *Drev;
    double *Fred, *temp2, *temp3;

    double *Mdense, *Lcrout, *Ucrout;

    // Finger related things
    // represents a single finger on the string
    struct Finger {
	double uf0, vf0;
	BreakpointFunction *force;
	BreakpointFunction *position;
	int xf_int_prev;
	double **IcLocs;
	double **JcLocs;

	// array of vectors. each one refers to a single location in Ic
	// and describes how it affects M. The vector contains pairs of values,
	// first an index of an element of M, then the index in Jc that it's multiplied
	// by
	vector<int> *IcM;

	// same for Jc
	vector<int> *JcM;
    };

    vector<Finger> fingers;

    // basic finger parameters (Mf is mass)
    double Mf, Kf, alphaf, betaf;

    // energy check stuff
    double *energy;
    CSRmatrix *P;
    double *uu1;
    double *etmp;
    double *eta;
};

#endif
