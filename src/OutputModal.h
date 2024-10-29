/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Represents an output from a modal component
 */
#ifndef _OUTPUTMODAL_H_
#define _OUTPUTMODAL_H_

#include "Output.h"
#include "ModalPlate.h"

class OutputModal : public Output {
 public:
    OutputModal(ModalPlate *comp, double pan, double x, double y);
    virtual ~OutputModal();

    virtual void runTimestep(int n);
    virtual void maybeMoveToGPU();

 protected:
    double *rp;
    int DIM;
    double h;
};


#endif
