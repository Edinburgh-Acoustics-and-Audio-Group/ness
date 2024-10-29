/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Implementation of the pluck_gen function
 */

#ifndef _MATLABFUNCTION_PLUCKGEN_H_
#define _MATLABFUNCTION_PLUCKGEN_H_

#include "MatlabFunction.h"

class MatlabFunction_PluckGen : public MatlabFunction {
 public:
    MatlabFunction_PluckGen();
    virtual ~MatlabFunction_PluckGen();

    virtual bool execute(MatlabCellContent *result);

 protected:
};

#endif
