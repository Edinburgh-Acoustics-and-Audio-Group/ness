/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * GPU-accelerated modal output
 */

#ifndef _GPU_OUTPUT_MODAL_H_
#define _GPU_OUTPUT_MODAL_H_

#include "GPUOutput.h"
#include "ModalPlate.h"

class GPUOutputModal : public GPUOutput {
 public:
    GPUOutputModal(ModalPlate *comp, double *rp);
    virtual ~GPUOutputModal();

    virtual void runTimestep(int n);

 protected:
    double *d_rp;
    int DIM;
    double h;
};


#endif
