/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "MatlabFunction_ClusterGen.h"
#include "Logger.h"
#include "InstrumentParserGuitar.h"

#include <cmath>
#include <cstdlib>
using namespace std;

MatlabFunction_ClusterGen::MatlabFunction_ClusterGen()
    : MatlabFunction("cluster_gen", 10, 11)
{
}

MatlabFunction_ClusterGen::~MatlabFunction_ClusterGen()
{
}

static double drand()
{
    return ((double)(rand())) / (double) RAND_MAX;
}

bool MatlabFunction_ClusterGen::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function cluster_gen");

    // check parameters
    if ((!checkParameterArray(0)) || (!checkParameterScalar(1)) || (!checkParameterScalar(2)) ||
	(!checkParameterScalar(3)) || (!checkParameterScalar(4)) || (!checkParameterScalar(5)) ||
	(!checkParameterScalar(6)) || (!checkParameterScalar(7)) || (!checkParameterScalar(8)) ||
	(!checkParameterScalar(9))) {
	return false;
    }

    int arrwidth = 5;

    int type = 1; // pluck by default
    if (paramCount == 11) {
	// type is passed in
	if (!checkParameterScalar(10)) return false;
	type = (int)params[10].scalar.value;
	arrwidth = 6;
    }

    if ((params[0].array.width != 0) && (params[0].array.width != 5) &&
	(params[0].array.width != 6)) {
	logMessage(3, "Invalid width for exc array passed to strum_gen");
	return false;
    }
    int oldheight = params[0].array.height;
    int oldwidth = params[0].array.width;
    if (params[9].array.width == 6) arrwidth = 6;

    // extract parameters
    double T = params[1].scalar.value;
    int N_pluck = (int)params[2].scalar.value;
    double dur = params[3].scalar.value;
    double amp = params[4].scalar.value;
    double amp_rand = params[5].scalar.value;
    double pluckdur = params[6].scalar.value;
    double pluckdur_rand = params[7].scalar.value;
    double pos = params[8].scalar.value;
    double pos_rand = params[9].scalar.value;

    int string_num = InstrumentParserGuitar::getNumStrings();

    // create output array
    result->type = CELL_ARRAY;
    result->array.width = arrwidth;
    result->array.height = N_pluck + oldheight;
    result->array.data = new double[arrwidth * (N_pluck + oldheight)];
    
    int i, j;

    // copy over the old plucks
    for (i = 0; i < oldheight; i++) {
	for (j = 0; j < oldwidth; j++) {
	    result->array.data[(i*arrwidth)+j] = params[0].array.data[(i*oldwidth)+j];
	}
	if (oldwidth < arrwidth) {
	    result->array.data[(i*arrwidth)+5] = 1; // pluck
	}
    }

    double *dat = &result->array.data[oldheight * arrwidth];

    // generate the plucks
    for (i = 0; i < N_pluck; i++) {
	// string index
	if (string_num) dat[i*arrwidth + 0] = (double)(rand() % string_num) + 1.0;
	else dat[i*arrwidth + 0] = 1.0;

	// start time
	dat[i*arrwidth + 1] = T + drand() * dur;

	// position
	double p = pos * (1.0 + pos_rand * (drand() - 0.5));
	if (p < 0.0) p = 0.0;
	if (p > 1.0) p = 1.0;
	dat[i*arrwidth + 2] = p;

	// duration
	dat[i*arrwidth + 3] = pluckdur * (1.0 + pluckdur_rand * (drand() - 0.5));

	// amplitude
	dat[i*arrwidth + 4] = amp * (1.0 + amp_rand * (drand() - 0.5));

	if (arrwidth > 5) {
	    dat[i*arrwidth + 5] = type;
	}
    }

    return true;
}
