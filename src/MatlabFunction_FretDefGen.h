/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Implementation of the fret_def_gen function
 */

#ifndef _MATLABFUNCTION_FRETDEFGEN_H_
#define _MATLABFUNCTION_FRETDEFGEN_H_

#include "MatlabFunction.h"

class MatlabFunction_FretDefGen : public MatlabFunction {
 public:
    MatlabFunction_FretDefGen();
    virtual ~MatlabFunction_FretDefGen();

    virtual bool execute(MatlabCellContent *result);

 protected:
};

#endif
