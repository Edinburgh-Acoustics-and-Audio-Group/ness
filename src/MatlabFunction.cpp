/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "MatlabFunction.h"
#include "Logger.h"

#include <cmath>

MatlabFunction::MatlabFunction()
{
    name = "";
    minParams = 0;
    maxParams = 0;
    params = NULL;
    paramCount = 0;
}

MatlabFunction::~MatlabFunction()
{
    if (params) delete[] params;
}


bool MatlabFunction::checkParameterScalar(int idx)
{
    if (params[idx].type != CELL_SCALAR) {
	logMessage(5, "Error: parameter %d of function %s must be a scalar value", idx+1, name.c_str());
	return false;
    }
    return true;
}

bool MatlabFunction::checkParameterArray(int idx)
{
    if (params[idx].type != CELL_ARRAY) {
	logMessage(5, "Error: parameter %d of function %s must be an array value", idx+1, name.c_str());
	return false;
    }
    return true;
}

bool MatlabFunction::checkParameter1DArray(int idx, int width)
{
    if ((params[idx].type != CELL_ARRAY) || (params[idx].array.height != 1)) {
	logMessage(5, "Error: parameter %d of function %s must be a 1D array value", idx+1, name.c_str());
	return false;
    }
    if (width >= 0) {
	if (params[idx].array.width != width) {
	    logMessage(5, "Error: parameter %d of function %s must be an array of size %d", idx+1,
		       name.c_str(), width);
	    return false;
	}
    }
    return true;
}

bool MatlabFunction::checkParameter2DArray(int idx, int width, int height)
{
    if (params[idx].type != CELL_ARRAY) {
	logMessage(5, "Error: parameter %d of function %s must be a 2D array value", idx+1, name.c_str());
	return false;
    }
    if (width >= 0) {
	if (params[idx].array.width != width) {
	    logMessage(5, "Error: parameter %d of function %s must be an array of width %d", idx+1,
		       name.c_str(), width);
	    return false;
	}
    }
    if (height >= 0) {
	if (params[idx].array.height != height) {
	    logMessage(5, "Error: parameter %d of function %s must be an array of height %d", idx+1,
		       name.c_str(), height);
	    return false;
	}
    }
    return true;
}



/*
 *
 * Sin function
 *
 */

MatlabFunction_Sin::MatlabFunction_Sin()
    : MatlabFunction("sin", 1, 1)
{
}

MatlabFunction_Sin::~MatlabFunction_Sin()
{
}

bool MatlabFunction_Sin::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function sin");

    // check parameters
    if (!checkParameterScalar(0)) {
	return false;
    }

    result->type = CELL_SCALAR;
    result->scalar.value = sin(params[0].scalar.value);
    return true;
}

/*
 *
 * Cos function
 *
 */

MatlabFunction_Cos::MatlabFunction_Cos()
    : MatlabFunction("cos", 1, 1)
{
}

MatlabFunction_Cos::~MatlabFunction_Cos()
{
}

bool MatlabFunction_Cos::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function sin");

    // check parameters
    if (!checkParameterScalar(0)) {
	return false;
    }

    result->type = CELL_SCALAR;
    result->scalar.value = cos(params[0].scalar.value);
    return true;
}

/*
 *
 * Exp function
 *
 */

MatlabFunction_Exp::MatlabFunction_Exp()
    : MatlabFunction("exp", 1, 1)
{
}

MatlabFunction_Exp::~MatlabFunction_Exp()
{
}

bool MatlabFunction_Exp::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function sin");

    // check parameters
    if (!checkParameterScalar(0)) {
	return false;
    }

    result->type = CELL_SCALAR;
    result->scalar.value = exp(params[0].scalar.value);
    return true;
}

/*
 *
 * Sqrt function
 *
 */

MatlabFunction_Sqrt::MatlabFunction_Sqrt()
    : MatlabFunction("sqrt", 1, 1)
{
}

MatlabFunction_Sqrt::~MatlabFunction_Sqrt()
{
}

bool MatlabFunction_Sqrt::execute(MatlabCellContent *result)
{
    logMessage(1, "Executing Matlab function sin");

    // check parameters
    if (!checkParameterScalar(0)) {
	return false;
    }

    result->type = CELL_SCALAR;
    result->scalar.value = sqrt(params[0].scalar.value);
    return true;
}

