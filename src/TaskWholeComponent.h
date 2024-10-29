/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * A parallel task that performs the entire update for one component
 */

#ifndef _TASKWHOLECOMPONENT_H_
#define _TASKWHOLECOMPONENT_H_

#include "Task.h"
#include "Component.h"

class TaskWholeComponent : public Task {
 public:
    TaskWholeComponent(Component *comp);
    virtual ~TaskWholeComponent();

    virtual void runTimestep(int n);

 protected:
    Component *component;
};

#endif
