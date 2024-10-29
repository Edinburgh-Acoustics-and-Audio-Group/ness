/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */

#include "Material.h"
#include "Logger.h"

Material::Material(string nm, double ym, double pr, double dy)
{
    logMessage(1, "Creating material: %s, %f, %f, %f", nm.c_str(), ym, pr, dy);
    name = nm;
    youngsModulus = ym;
    poissonsRatio = pr;
    density = dy;
}

Material::~Material()
{
}

