/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Implementation of the string_def_gen function
 */

#ifndef _MATLABFUNCTION_STRINGDEFGEN_H_
#define _MATLABFUNCTION_STRINGDEFGEN_H_

#include "MatlabFunction.h"

class MatlabFunction_StringDefGen : public MatlabFunction {
 public:
    MatlabFunction_StringDefGen();
    virtual ~MatlabFunction_StringDefGen();

    virtual bool execute(MatlabCellContent *result);

 protected:
};

#endif
