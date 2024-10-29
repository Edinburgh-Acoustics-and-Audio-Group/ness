/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * GPU acceleration for modal plate components
 */
#ifndef _GPU_MODAL_PLATE_H_
#define _GPU_MODAL_PLATE_H_

class GPUModalPlate {
 public:
    GPUModalPlate(int A, int DIM, double *H1, double *C, double *C1, double *C2,
		  double **u, double **u1, double **u2);
    virtual ~GPUModalPlate();

    void runTimestep(int n, double *u, double *u1, double *u2);

    bool isOK() { return ok; }

 protected:
    int A, DIM;

    int grid1, block1;
    int grid2, block2;
    int grid3, block3;

    double *d_u, *d_u1, *d_u2;

    double *d_H1;
    double *d_C;
    double *d_C1;
    double *d_C2;
    double *d_G;
    double *d_t1;
    double *d_t2;

    bool ok;
};

#endif
