/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */
#include "Component3D.h"

#include <cmath>
#include <cstring>
using namespace std;

Component3D::Component3D(string name) : Component(name)
{
}

Component3D::~Component3D()
{
}

int Component3D::getIndex(int x, int y, int z)
{
    return (x * ny) + y + (z * nxny);
}

int Component3D::getIndexf(double x, double y, double z)
{
    return getIndex((int)floor(x * ((double)nx)), (int)floor(y * ((double)ny)),
		    (int)floor(z * ((double)nz)));
}

void Component3D::getInterpolationInfo(InterpolationInfo *info, double x, double y, double z)
{
    info->type = INTERPOLATION_TRILINEAR;
    info->idx = getIndexf(x, y, z);
    info->nx = ny;
    info->nxny = nxny;
    double ax = (x * ((double)nx)) - floor(x * ((double)nx));
    double ay = (y * ((double)ny)) - floor(y * ((double)ny));
    double az = (z * ((double)nz)) - floor(z * ((double)nz));
    info->alpha[0] = (1.0 - ax) * (1.0 - ay) * (1.0 - az);
    info->alpha[1] = (1.0 - ax) * ay * (1.0 - az);
    info->alpha[2] = ax * (1.0 - ay) * (1.0 - az);
    info->alpha[3] = ax * ay * (1.0 - az);
    info->alpha[4] = (1.0 - ax) * (1.0 - ay) * az;
    info->alpha[5] = (1.0 - ax) * ay * az;
    info->alpha[6] = ax * (1.0 - ay) * az;
    info->alpha[7] = ax * ay * az;

    if ((x >= 0.9999) || (y >= 0.9999) || (z >= 0.9999)) {
	// don't interpolate furthest points as might end up reading off end of array
	info->type = INTERPOLATION_NONE;
    }
}

void Component3D::allocateState(int nx, int ny, int nz)
{
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    nxny = nx * ny;
    ss = nxny * nz;
    u = new double[ss + 2*nxny]; // add a halo at either end
    u1 = new double[ss + 2*nxny];
    u2 = new double[ss + 2*nxny];

    initialiseState(ss + 2*nxny);
}
