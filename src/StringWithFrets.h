/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * String component with frets
 */
#ifndef _STRINGWITHFRETS_H_
#define _STRINGWITHFRETS_H_

#include "ComponentString.h"

extern "C" {
#include "csrmatrix.h"
#include "pcg.h"
};

class StringWithFrets : public ComponentString {
 public:
    StringWithFrets(string name, double L, double rho, double T, double E, double r,
		    double T60_0, double T60_1000, double xc1, double yc1, double xc2,
		    double yc2, int numfrets, double fretheight,
		    double backboardheight, double backboardvar);
    virtual ~StringWithFrets();

    virtual void runTimestep(int n);

    void setParams(double K, double alpha, double iter) {
	this->K = K;
	this->alphaNewton = alpha;
	this->iter = iter;
    }

    virtual void logMatrices();

 protected:
    void newtonsMethod();

    CSRmatrix *Ic;
    CSRmatrix *Jc;
    CSRmatrix *M;
    CSRmatrix *J;

    double *btot;
    double *a;
    double *b;
    double *r;
    double *R;

    double K, alphaNewton, iter;

    double *F;
    double *temp;
    double *utmp;

    pcg_info_t *pcg;
};

#endif
