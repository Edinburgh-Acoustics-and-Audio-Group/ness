/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * A component embedded within an airbox
 */
#ifndef _EMBEDDING_H_
#define _EMBEDDING_H_

#include "Connection.h"
#include "PlateEmbedded.h"
#include "Airbox.h"

extern "C" {
#include "csrmatrix.h"
};

#ifdef USE_GPU
#include "GPUEmbedding.h"
#endif

class Embedding : public Connection {
 public:
    Embedding(Airbox *airbox, PlateEmbedded *plate);
    virtual ~Embedding();

    virtual void runTimestep(int n);

    virtual void maybeMoveToGPU();

 protected:
    // forward and reverse interpolation matrices
    CSRmatrix *IMat, *JMat;

    // prescaled IMat
    CSRmatrix *BfIMat;

    double Bf;

    int Diff_size;
    int pd, pu;

    double *Diff_np, *Diff_n, *Diff_nm;
    double *Sum_np;
    double *Diffstar_n, *Diffstar_nm;
    double *Diff_tmp;

    double *true_Psi;

    double *transferBuffer;

#ifdef USE_GPU
    GPUEmbedding *gpuEmbedding;
#endif
};


#endif
