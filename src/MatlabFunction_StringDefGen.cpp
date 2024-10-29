/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "MatlabFunction_StringDefGen.h"
#include "Logger.h"

#include <cmath>
#include <cstdlib>
using namespace std;

#ifndef M_PI
#define M_PI 3.1415926536
#endif

MatlabFunction_StringDefGen::MatlabFunction_StringDefGen()
    : MatlabFunction("string_def_gen", 7, 7)
{
}

MatlabFunction_StringDefGen::~MatlabFunction_StringDefGen()
{
}

bool MatlabFunction_StringDefGen::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function string_def_gen");

    // check parameters
    if (!checkParameter2DArray(0, 2)) return false;
    if (!checkParameter1DArray(1)) return false;
    if (!checkParameter1DArray(2)) return false;
    if (!checkParameter1DArray(3)) return false;
    if (!checkParameterScalar(4)) return false;
    if (!checkParameter1DArray(5)) return false;
    if (!checkParameter1DArray(6)) return false;

    // work out number of strings
    int string_num = params[1].array.width;

    // check it matches the other per-string parameters
    if ((params[2].array.width != string_num) ||
	(params[3].array.width != string_num) ||
	(params[5].array.width != string_num) ||
	(params[6].array.width != string_num)) {
	logMessage(5, "Error: array sizes must match for string_def_gen");
	return false;
    }

    // allocate the output array
    result->type = CELL_ARRAY;
    result->array.width = 7;
    result->array.height = string_num;
    result->array.data = new double[7 * string_num];

    int i;
    // generate the data for each string
    double L = params[4].scalar.value;
    for (i = 0; i < string_num; i++) {
	// handle material
	int matidx = ((int)params[2].array.data[i]) - 1;
	double E = params[0].array.data[(matidx * 2) + 1];
	double rho = params[0].array.data[(matidx * 2) + 0];

	// compute frequency
	double note = params[1].array.data[i];
	double freq = 261.6 * pow(2.0, note / 12.0);

	double inharmonicity = params[3].array.data[i];

	// compute radius
	double r = ((4.0 * L * L) / M_PI) * freq * sqrt(rho * inharmonicity / E);

	// compute area
	double A = M_PI * r * r;

	// compute tension
	double T = (4.0 * L * L) * rho * A * freq * freq;

	// fetch loss parameters
	double T60_0 = params[5].array.data[i];
	double T60_1000 = params[6].array.data[i];

	result->array.data[7*i + 0] = L; // length
	result->array.data[7*i + 1] = E;
	result->array.data[7*i + 2] = T;
	result->array.data[7*i + 3] = r;
	result->array.data[7*i + 4] = rho;
	result->array.data[7*i + 5] = T60_0;
	result->array.data[7*i + 6] = T60_1000;
    }

    return true;
}

