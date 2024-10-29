/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Implementation of the strum_gen_multi function
 */

#ifndef _MATLABFUNCTION_STRUMGENMULTI_H_
#define _MATLABFUNCTION_STRUMGENMULTI_H_

#include "MatlabFunction.h"

class MatlabFunction_StrumGenMulti : public MatlabFunction {
 public:
    MatlabFunction_StrumGenMulti();
    virtual ~MatlabFunction_StrumGenMulti();

    virtual bool execute(MatlabCellContent *result);

 protected:
};

#endif
