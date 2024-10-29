/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "MatlabFunction_StrumGen.h"
#include "Logger.h"
#include "InstrumentParserGuitar.h"

#include <cstdlib>
using namespace std;

MatlabFunction_StrumGen::MatlabFunction_StrumGen()
    : MatlabFunction("strum_gen", 11, 12)
{
}

MatlabFunction_StrumGen::~MatlabFunction_StrumGen()
{
}

static double drand()
{
    return ((double)(rand())) / (double) RAND_MAX;
}

bool MatlabFunction_StrumGen::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function strum_gen");

    // check parameters
    if ((!checkParameterArray(0)) || (!checkParameterScalar(1)) || (!checkParameterScalar(2)) ||
	(!checkParameterScalar(3)) || (!checkParameterScalar(4)) || (!checkParameterScalar(5)) ||
	(!checkParameterScalar(6)) || (!checkParameterScalar(7)) || (!checkParameterScalar(8)) ||
	(!checkParameterScalar(9)) || (!checkParameterScalar(10))) {
	return false;
    }

    int arrwidth = 5;
    int oldwidth = params[0].array.width;
    int type = 1;
    bool newstyle = false;

    if (paramCount == 12) {
	arrwidth = 6;
	if (!checkParameterScalar(11)) return false;
	type = (int)params[11].scalar.value;
	newstyle = true;
    }

    if (oldwidth == 6) arrwidth = 6;
    if ((oldwidth != 0) && (oldwidth != 5) && (oldwidth != 6)) {
	logMessage(3, "Invalid width for exc array passed to strum_gen");
	return false;
    }
    int oldheight = params[0].array.height;

    // extract parameters
    double T = params[1].scalar.value;
    double dur = params[2].scalar.value;
    int ud = (int)params[3].scalar.value;
    double amp = params[4].scalar.value;
    double amp_rand = params[5].scalar.value;
    double pluckdur = params[6].scalar.value;
    double pluckdur_rand = params[7].scalar.value;
    double pos = params[8].scalar.value;
    double pos_rand = params[9].scalar.value;
    double times_rand = params[10].scalar.value;

    int string_num = InstrumentParserGuitar::getNumStrings();

    // allocate result
    result->type = CELL_ARRAY;
    result->array.width = arrwidth;
    result->array.height = string_num + oldheight;
    result->array.data = new double[arrwidth * (string_num + oldheight)];

    int i, j;

    // copy over the old plucks
    for (i = 0; i < (oldheight); i++) {
	for (j = 0; j < oldwidth; j++) {
	    result->array.data[i*arrwidth + j] = params[0].array.data[i*oldwidth + j];
	}
	if (oldwidth < arrwidth) {
	    result->array.data[i*arrwidth + 5] = type;
	}
    }

    double *dat = &result->array.data[oldheight * arrwidth];

    // generate the plucks
    for (i = 0; i < string_num; i++) {
	// string index
	if (ud) {
	    dat[i*arrwidth + 0] = (double)(string_num - i);
	}
	else {
	    dat[i*arrwidth + 0] = (double)(i+1);
	}

	// start time
	double tm = T + (((double)i) / (double)(string_num-1)) * dur;
	if (newstyle) {
	    // net code version works differently
	    tm = tm + times_rand * dur * (drand() - 0.5);
	}
	else {
	    tm = tm * (1.0 + times_rand * (drand() - 0.5));
	}
	if (tm < 0.0) tm = 0.0;
	dat[i*arrwidth + 1] = tm;

	// position
	double p = pos * (1.0 + pos_rand * (drand() - 0.5));
	if (p < 0.0) p = 0.0;
	if (p > 1.0) p = 1.0;
	dat[i*arrwidth + 2] = p;

	// duration
	dat[i*arrwidth + 3] = pluckdur * (1.0 + pluckdur_rand * (drand() - 0.5));

	// amplitude
	dat[i*arrwidth + 4] = amp * (1.0 + amp_rand * (drand()-0.5));
    }

    return true;
}
