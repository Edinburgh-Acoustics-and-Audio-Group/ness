/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TaskWholeComponent.h"

TaskWholeComponent::TaskWholeComponent(Component *comp)
{
    component = comp;
}

TaskWholeComponent::~TaskWholeComponent()
{
}

void TaskWholeComponent::runTimestep(int n)
{
    component->runTimestep(n);
}

