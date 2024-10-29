/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */
#include "Component2D.h"

#include <cstring>
#include <cmath>
using namespace std;

Component2D::Component2D(string name) : Component(name)
{
}

Component2D::~Component2D()
{
}

int Component2D::getIndex(int x, int y, int z)
{
    return (x * ny) + y;
}

int Component2D::getIndexf(double x, double y, double z)
{
    return getIndex((int)floor(x * ((double)nx)), (int)floor(y * ((double)ny)));
}

void Component2D::getInterpolationInfo(InterpolationInfo *info, double x, double y, double z)
{
    info->type = INTERPOLATION_BILINEAR;
    info->idx = getIndexf(x, y, z);
    info->nx = ny;
    info->nxny = 0;
    double ax = (x * ((double)nx)) - floor(x * ((double)nx));
    double ay = (y * ((double)ny)) - floor(y * ((double)ny));
    info->alpha[0] = (1.0 - ay) * (1.0 - ax);
    info->alpha[1] = ay * (1.0 - ax);
    info->alpha[2] = (1.0 - ay) * ax;
    info->alpha[3] = ay * ax;
    info->alpha[4] = 0.0;
    info->alpha[5] = 0.0;
    info->alpha[6] = 0.0;
    info->alpha[7] = 0.0;

    if ((x >= 0.9999) || (y >= 0.9999)) {
	// don't interpolate furthest points as might end up reading off end of array
	info->type = INTERPOLATION_NONE;
    }
}

void Component2D::allocateState(int nx, int ny)
{
    this->nx = nx;
    this->ny = ny;
    ss = nx * ny;
    u = new double[ss];
    u1 = new double[ss];
    u2 = new double[ss];
    initialiseState(ss);
}
