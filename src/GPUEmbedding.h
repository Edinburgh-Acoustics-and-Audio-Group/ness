/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * GPU implementation of Embedding
 */

#ifndef _GPU_EMBEDDING_H_
#define _GPU_EMBEDDING_H_

#include "Airbox.h"

extern "C" {
#include "csrmatrix.h"
};

#include "GPUUtil.h"

class GPUEmbedding {
 public:
    GPUEmbedding(CSRmatrix *BfIMat, CSRmatrix *JMat, int Diff_size, int pd, int pu, Airbox *airbox,
		 double k, int plateSS, double *h_true_Psi);
    virtual ~GPUEmbedding();

    void runTimestep(int n, Airbox *airbox, double *hostBuffer);

 protected:
    GpuMatrix_t *JMat;
    GpuMatrix_t *BfIMat;
    
    double *Sum_np;
    double *Diff_np, *Diff_n, *Diff_nm;
    double *Diffstar_n, *Diffstar_nm;
    double *Diff_tmp;
    double *transferBuffer;
    double *true_Psi;

    int Diff_size;
    int plateSS;
    int pd, pu;

    double Qdivk, Gammasq, lambdasq, diffnfac;

    int gridSize;
};

#endif
