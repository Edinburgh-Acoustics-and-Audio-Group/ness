/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "MatlabFunction_FretDefGen.h"
#include "Logger.h"

#include <cmath>
using namespace std;

MatlabFunction_FretDefGen::MatlabFunction_FretDefGen()
    : MatlabFunction("fret_def_gen", 3, 4)
{
}

MatlabFunction_FretDefGen::~MatlabFunction_FretDefGen()
{
}

bool MatlabFunction_FretDefGen::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function fret_def_gen, paramCount=%d", paramCount);

    // check parameters
    int p = 0;
    MatlabArray *oldfrets = NULL;
    int oldheight = 0;
    if (paramCount == 4) {
	if (!checkParameterArray(0)) return false;
	oldfrets = &params[0].array;
	if ((oldfrets->width != 0) && (oldfrets->width != 2)) {
	    logMessage(3, "Invalid width for frets array passed to fret_def_gen");
	    return false;
	}
	oldheight = oldfrets->height;
	p++;
    }

    if (!checkParameterScalar(p)) return false;
    if (!checkParameterScalar(p+1)) return false;
    if (!checkParameterScalar(p+2)) return false;

    int fretnum = (int)params[p].scalar.value;
    double wid = params[p+1].scalar.value;
    double height = params[p+2].scalar.value;

    // allocate the output array
    result->type = CELL_ARRAY;
    result->array.width = 2;
    result->array.height = fretnum + oldheight;
    result->array.data = new double[2 * (fretnum + oldheight)];

    int i;
    // copy over old frets
    if (oldheight) {
	for (i = 0; i < (oldheight * 2); i++) {
	    result->array.data[i] = oldfrets->data[i];
	}
    }

    // generate the data for each fret
    double *dat = &result->array.data[oldheight * 2];
    for (i = 0; i < fretnum; i++) {
	double pos = 1.0 - pow(2.0, (-wid * ((double)(i+1)) / 12.0));
	dat[0] = pos;
	dat[1] = height;
	dat += 2;
    }

    return true;
}
