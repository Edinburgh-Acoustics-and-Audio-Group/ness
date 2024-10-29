/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Implementation of the cluster_gen function
 */

#ifndef _MATLABFUNCTION_CLUSTERGEN_H_
#define _MATLABFUNCTION_CLUSTERGEN_H_

#include "MatlabFunction.h"

class MatlabFunction_ClusterGen : public MatlabFunction {
 public:
    MatlabFunction_ClusterGen();
    virtual ~MatlabFunction_ClusterGen();

    virtual bool execute(MatlabCellContent *result);

 protected:
};

#endif
