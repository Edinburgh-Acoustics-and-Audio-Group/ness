/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU acceleration for Plate components
 */
#ifndef _GPU_PLATE_H_
#define _GPU_PLATE_H_

extern "C" {
#include "csrmatrix.h"
};

class GPUPlate {
 public:
    GPUPlate(int nx, int ny, CSRmatrix *B, CSRmatrix *C, double **u,
	     double **u1, double **u2);
    virtual ~GPUPlate();

    void runTimestep(int n, double *u, double *u1, double *u2);

    bool isOK() { return ok; }

 protected:
    int nx, ny;
    unsigned char *indexb, *indexc;
    double *coeffsb, *coeffsc;

    int gridW, gridH;

    double *d_u, *d_u1, *d_u2;

    bool ok;
};

#endif
