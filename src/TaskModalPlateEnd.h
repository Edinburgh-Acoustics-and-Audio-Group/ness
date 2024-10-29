/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * A serial task that finishes the timestep update for the modal plate
 */

#ifndef _TASKMODALPLATEEND_H_
#define _TASKMODALPLATEEND_H_

#include "Task.h"
#include "ModalPlate.h"

class TaskModalPlateEnd : public Task {
 public:
    TaskModalPlateEnd(ModalPlate *mp);
    virtual ~TaskModalPlateEnd();

    virtual void runTimestep(int n);

 protected:
    ModalPlate *modalPlate;
};

#endif
