/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * GPU accelerated version of indexed airbox (including viscous airbox)
 */

#ifndef _GPU_AIRBOX_INDEXED_H_
#define _GPU_AIRBOX_INDEXED_H_

class GPUAirboxIndexed {
 public:
    GPUAirboxIndexed(int nx, int ny, int nz, double **u, double **u1,
		     double **u2, unsigned char *h_index, double *h_coeffs);
    virtual ~GPUAirboxIndexed();

    virtual void runTimestep(int n, double *u, double *u1, double *u2);
    virtual void runPartialUpdate(double *u, double *u1, double *u2, int start, int len);

    bool isOK() { return ok; }

 protected:
    double *d_u, *d_u1, *d_u2;

    int nx, ny, nz;
    int gridW, gridH, gridD;

    unsigned char *index;
    double *coeffs;

    bool ok;
};

#endif
