/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Fretboard component
 */
#ifndef _FRETBOARD_H_
#define _FRETBOARD_H_

#include "Component1D.h"
#include "MathUtil.h"

extern "C" {
#include "csrmatrix.h"
};

#include <vector>
using namespace std;

class Fretboard : public Component1D {
 public:
    Fretboard(string name, double L, double E, double T, double r, double rho, double T60_0,
	      double T60_1000, int fretnum, double b0, double b1, double fretheight, double Kb,
	      double alphab, double betab, int itnum);
    virtual ~Fretboard();

    virtual void runTimestep(int n);

    virtual void logMatrices();

    void setFingerParams(double Mf, double Kf, double alphaf, double betaf, double uf0, double vf0,
			 vector<double> *fingertime, vector<double> *fingerpos,
			 vector<double> *fingerforce);

 protected:
    CSRmatrix *extendMatrix(CSRmatrix *old, double val);

    CSRmatrix *removeDiagonalEntries(CSRmatrix *mat);
    void matrixMultiplyReuse(CSRmatrix *in1, CSRmatrix *in2, CSRmatrix *out, double *diag);

    double L;
    double E;
    double T;
    double radius;
    double rho;
    double T60_0;
    double T60_1000;

    double fac, A, h;

    // main update matrices for string
    CSRmatrix *B, *C;

    // matrices for Newton solver
    CSRmatrix *I0, *J0, *M0;

    // fret-related stuff
    int Nb;
    double *b;

    // finger position and force  at each timestep
    double *ff;
    int *xf_int;
    double *xf_frac;

    // vectors for Newton
    double *r, *eta1, *eta2, *g, *q, *K, *alphan, *beta;
    int itnum;

    double *phi_ra, *Mdiag, *R;

    newton_solver_t *newton;

};

#endif
