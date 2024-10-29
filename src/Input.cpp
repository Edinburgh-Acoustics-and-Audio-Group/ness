/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#include "Input.h"
#include "GlobalSettings.h"

#include <cmath>
using namespace std;

Input::Input(Component *comp, double x, double y, double z)
{
    component = comp;
    index = comp->getIndexf(x, y, z);
}

Input::Input(Component *comp)
{
    component = comp;
}

Input::~Input()
{
}

int Input::timeToTimestep(double tm)
{
    return (int)floor(tm * GlobalSettings::getInstance()->getSampleRate());
}

int Input::firstInputTimestep = 1000000000;
