/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Implementation of the strum_gen function
 */

#ifndef _MATLABFUNCTION_STRUMGEN_H_
#define _MATLABFUNCTION_STRUMGEN_H_

#include "MatlabFunction.h"

class MatlabFunction_StrumGen : public MatlabFunction {
 public:
    MatlabFunction_StrumGen();
    virtual ~MatlabFunction_StrumGen();

    virtual bool execute(MatlabCellContent *result);

 protected:
};

#endif
