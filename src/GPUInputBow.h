/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU-accelerated bow input
 */

#ifndef _GPU_INPUT_BOW_H_
#define _GPU_INPUT_BOW_H_

class GPUInputBow {
 public:
    GPUInputBow(int len, double *velocity, double *force, double k, double jb, int itnum,
		double sigma, int index);
    virtual ~GPUInputBow();

    void runTimestep(int n, double *s, double *s2);

 protected:
    double *velocity;
    double *force;
    double k;
    double jb;
    int itnum;
    double sigma;
    int index;
};

#endif
