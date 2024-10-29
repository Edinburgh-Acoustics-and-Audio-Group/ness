/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * A parallel task that updates one row-type for the modal plate
 */

#ifndef _TASKMODALPLATEROW_H_
#define _TASKMODALPLATEROW_H_

#include "Task.h"
#include "ModalPlate.h"

class TaskModalPlateRow : public Task {
 public:
    TaskModalPlateRow(ModalPlate *mp, int row);
    virtual ~TaskModalPlateRow();

    virtual void runTimestep(int n);

 protected:
    ModalPlate *modalPlate;
    int rowType;
};

#endif
