/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TaskModalPlateEnd.h"

TaskModalPlateEnd::TaskModalPlateEnd(ModalPlate *mp)
{
    modalPlate = mp;
}

TaskModalPlateEnd::~TaskModalPlateEnd()
{
}

void TaskModalPlateEnd::runTimestep(int n)
{
    modalPlate->finishUpdate(n);
}
