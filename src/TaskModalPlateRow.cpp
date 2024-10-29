/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TaskModalPlateRow.h"

TaskModalPlateRow::TaskModalPlateRow(ModalPlate *mp, int row)
{
    modalPlate = mp;
    rowType = row;
}

TaskModalPlateRow::~TaskModalPlateRow()
{
}

void TaskModalPlateRow::runTimestep(int n)
{
    modalPlate->runRowType(n, rowType);
}
