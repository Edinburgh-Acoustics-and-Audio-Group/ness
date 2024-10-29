/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Represents a function that can be called from Matlab instrument/score files.
 */

#ifndef _MATLABFUNCTION_H_
#define _MATLABFUNCTION_H_

#include "MatlabParser.h"

#include <string>
#include <vector>
using namespace std;

class MatlabFunction {
 public:
    MatlabFunction();
    MatlabFunction(string name, int minParams, int maxParams) {
	this->name = name;
	this->minParams = minParams;
	this->maxParams = maxParams;
	params = new MatlabCellContent[maxParams];
	paramCount = 0;
    }
    virtual ~MatlabFunction();

    string getName() { return name; }
    int getMinParams() { return minParams; }
    int getMaxParams() { return maxParams; }

    void setParameter(int idx, MatlabCellContent val) {
	if (idx >= maxParams) return;
	params[idx] = val;

	if ((idx+1) > paramCount) paramCount = idx + 1;
    }

    virtual bool execute(MatlabCellContent *result) = 0;

 protected:
    bool checkParameterScalar(int idx);
    bool checkParameterArray(int idx);
    bool checkParameter1DArray(int idx, int width = -1);
    bool checkParameter2DArray(int idx, int width = -1, int height = -1);

    string name;
    int minParams, maxParams;
    MatlabCellContent *params;
    int paramCount;
};

class MatlabFunction_Sin : public MatlabFunction {
 public:
    MatlabFunction_Sin();
    virtual ~MatlabFunction_Sin();
    virtual bool execute(MatlabCellContent *result);
};

class MatlabFunction_Cos : public MatlabFunction {
 public:
    MatlabFunction_Cos();
    virtual ~MatlabFunction_Cos();
    virtual bool execute(MatlabCellContent *result);
};

class MatlabFunction_Exp : public MatlabFunction {
 public:
    MatlabFunction_Exp();
    virtual ~MatlabFunction_Exp();
    virtual bool execute(MatlabCellContent *result);
};

class MatlabFunction_Sqrt : public MatlabFunction {
 public:
    MatlabFunction_Sqrt();
    virtual ~MatlabFunction_Sqrt();
    virtual bool execute(MatlabCellContent *result);
};

#endif
