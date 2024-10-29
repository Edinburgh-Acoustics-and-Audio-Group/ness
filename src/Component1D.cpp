/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */
#include "Component1D.h"
#include "SettingsManager.h"

#include <cmath>
#include <cstring>
using namespace std;

Component1D::Component1D(string name) : Component(name)
{
}

Component1D::~Component1D()
{
}

int Component1D::getIndex(int x, int y, int z)
{
    if (x == ss) x--;
    return x;
}

int Component1D::getIndexf(double x, double y, double z)
{
    return getIndex((int)floor(x * ((double)ss)));
}

void Component1D::getInterpolationInfo(InterpolationInfo *info, double x, double y, double z)
{
    info->type = INTERPOLATION_LINEAR;
    info->idx = getIndexf(x, y, z);
    info->nx = 0;
    info->nxny = 0;
    double ax = (x * ((double)ss)) - floor(x * ((double)ss));
    info->alpha[0] = 1.0 - ax;
    info->alpha[1] = ax;
    info->alpha[2] = 0.0;
    info->alpha[3] = 0.0;
    info->alpha[4] = 0.0;
    info->alpha[5] = 0.0;
    info->alpha[6] = 0.0;
    info->alpha[7] = 0.0;

    if (x >= 0.9999) {
	// don't interpolate far RHS point as might end up reading off end of array
	info->type = INTERPOLATION_NONE;
    }
}

void Component1D::allocateState(int n)
{
    ss = n;
    u = new double[ss];
    u1 = new double[ss];
    u2 = new double[ss];
    initialiseState(ss);
}

