/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "MatlabFunction_StrumGenMulti.h"
#include "Logger.h"
#include "MathUtil.h"
#include "InstrumentParserGuitar.h"

#include <cstdlib>
using namespace std;

MatlabFunction_StrumGenMulti::MatlabFunction_StrumGenMulti()
    : MatlabFunction("strum_gen_multi", 12, 12)
{
}

MatlabFunction_StrumGenMulti::~MatlabFunction_StrumGenMulti()
{
}

static double drand()
{
    return ((double)(rand())) / (double) RAND_MAX;
}

bool MatlabFunction_StrumGenMulti::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function strum_gen_multi");

    // check parameters
    if ((!checkParameterArray(0)) || (!checkParameterArray(1)) || (!checkParameterArray(2)) ||
	(!checkParameterArray(3)) || (!checkParameterScalar(4)) || (!checkParameterArray(5)) ||
	(!checkParameterScalar(6)) || (!checkParameterArray(7)) || (!checkParameterScalar(8)) ||
	(!checkParameterArray(9)) || (!checkParameterScalar(10)) || (!checkParameterScalar(11))) {
	return false;
    }

    if ((params[0].array.width != 0) && (params[0].array.width != 5)) {
	logMessage(3, "Invalid width for exc array passed to strum_gen_multi");
	return false;
    }
    int oldheight = params[0].array.height;

    // extract parameters
    MatlabArray tax = params[1].array;
    MatlabArray density = params[2].array;
    MatlabArray dur = params[3].array;
    int ud = (int)params[4].scalar.value;
    MatlabArray amp = params[5].array;
    double amp_rand = params[6].scalar.value;
    MatlabArray pluckdur = params[7].array;
    double pluckdur_rand = params[8].scalar.value;
    MatlabArray pos = params[9].array;
    double pos_rand = params[10].scalar.value;
    double times_rand = params[11].scalar.value;

    int string_num = InstrumentParserGuitar::getNumStrings();

    // check that array sizes match
    if ((tax.height > 1) || (density.height > 1) || (dur.height > 1) ||
	(amp.height > 1) || (pluckdur.height > 1) || (pos.height > 1)) {
	logMessage(3, "Arrays passed to strum_gen_multi must be one-dimensional");
	return false;
    }
    if ((tax.width != density.width) || (dur.width != density.width) ||
	(amp.width != density.width) || (pluckdur.width != density.width) ||
	(pos.width != density.width)) {
	logMessage(3, "Arrays passed to strum_gen_multi must be same size");
	return false;
    }

    // generate T array
    vector<double> T;
    T.push_back(tax.data[0]);
    double taxend = tax.data[tax.width - 1];
    double Tnext = tax.data[0];
    double Tnextnext;
    while (Tnext <= taxend) {
	interp1(tax.data, density.data, &Tnext, tax.width, 1, &Tnextnext);
	Tnext = Tnext + 1.0 / Tnextnext;
	if (Tnext < taxend) T.push_back(Tnext);
    }

    // allocate result array
    int i, j;
    int strum_num = T.size();
    int newtot = strum_num * string_num;
    result->type = CELL_ARRAY;
    result->array.width = 5;
    result->array.height = oldheight + newtot;
    result->array.data = new double[5 * (oldheight + newtot)];

    // copy over the old plucks
    for (i = 0; i < (oldheight*5); i++) {
	result->array.data[i] = params[0].array.data[i];
    }
    double *dat = &result->array.data[oldheight * 5];

    // generate the strums
    int currdir = ud;
    if (ud == 2) currdir = 1;

    for (i = 0; i < strum_num; i++) {
	double Tcur = T[i];
	double durcur, poscur, pluckdurcur, ampcur;
	interp1(tax.data, dur.data, &Tcur, tax.width, 1, &durcur);
	interp1(tax.data, pos.data, &Tcur, tax.width, 1, &poscur);
	interp1(tax.data, pluckdur.data, &Tcur, tax.width, 1, &pluckdurcur);
	interp1(tax.data, amp.data, &Tcur, tax.width, 1, &ampcur);

	for (j = 0; j < string_num; j++) {
	    if (currdir == 0) {
		dat[0] = (double)(j+1);
	    }
	    else {
		dat[0] = (double)(string_num - j);
	    }

	    double tm = Tcur + (((double)j) / ((double)(string_num-1))) * durcur;
	    tm = tm + times_rand * durcur * (drand() - 0.5);
	    if (tm < 0.0) tm = 0.0;
	    dat[1] = tm;

	    double pos = poscur * (1.0 + pos_rand * (drand() - 0.5));
	    if (pos < 0.0) pos = 0.0;
	    if (pos > 1.0) pos = 1.0;
	    dat[2] = pos;

	    dat[3] = pluckdurcur * (1.0 + pluckdur_rand * (drand() - 0.5));

	    dat[4] = ampcur * (1.0 + amp_rand * (drand() - 0.5));

	    dat += 5;
	}

	if (ud == 2) currdir = 1 - currdir;
    }

    return true;
}
