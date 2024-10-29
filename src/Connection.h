/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Represents any sort of connection between two components.
 */
#ifndef _CONNECTION_H_
#define _CONNECTION_H_

#include "Component.h"
#include "Task.h"

class Connection {
 public:
    Connection(Component *c1, Component *c2);
    virtual ~Connection();

    virtual void runTimestep(int n) = 0;

    virtual void maybeMoveToGPU();

    virtual double *getEnergy();

    virtual void getParallelTasks(vector<Task*> &tasks);
    virtual void getSerialTasks(vector<Task*> &tasks);

 protected:
    Component *c1;
    Component *c2;
};

#endif
