/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "MatlabFunction_PluckGen.h"
#include "Logger.h"

#include <cstdlib>
using namespace std;

MatlabFunction_PluckGen::MatlabFunction_PluckGen()
    : MatlabFunction("pluck_gen", 6, 7)
{
}

MatlabFunction_PluckGen::~MatlabFunction_PluckGen()
{
}

bool MatlabFunction_PluckGen::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function pluck_gen");

    // check parameters
    if ((!checkParameterArray(0)) || (!checkParameterScalar(1)) || (!checkParameterScalar(2)) ||
	(!checkParameterScalar(3)) || (!checkParameterScalar(4)) || (!checkParameterScalar(5))) {
	return false;
    }

    int oldwidth = params[0].array.width;
    int arrwidth = 5;
    int type = 1;

    if (paramCount == 7) {
	if (!checkParameterScalar(6)) return false;
	arrwidth = 6;
	type = (int)params[6].scalar.value;
    }

    if ((oldwidth != 0) && (oldwidth != 5) && (oldwidth != 6)) {
	logMessage(3, "Invalid width for exc array passed to pluck_gen");
	return false;
    }
    if (oldwidth == 6) arrwidth = 6;
    if (oldwidth == 0) oldwidth = 5;

    int oldheight = params[0].array.height;

    // allocate result
    result->type = CELL_ARRAY;
    result->array.width = arrwidth;
    result->array.height = 1 + oldheight;
    result->array.data = new double[arrwidth * (1 + oldheight)];

    int i, j;

    // copy over the old plucks
    for (i = 0; i < oldheight; i++) {
	for (j = 0; j < oldwidth; j++) {
	    result->array.data[(i*arrwidth)+j] = params[0].array.data[(i*oldwidth)+j];
	}
	if (oldwidth < arrwidth) result->array.data[(i*arrwidth)+5] = 1.0;
    }

    double *dat = &result->array.data[oldheight * arrwidth];

    // generate the new pluck
    dat[0] = params[1].scalar.value;
    dat[1] = params[2].scalar.value;
    dat[2] = params[3].scalar.value;
    dat[3] = params[4].scalar.value;
    dat[4] = params[5].scalar.value;

    if (arrwidth > 5) dat[5] = type;

    return true;
}
