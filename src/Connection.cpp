/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */
#include "Connection.h"
#include "TaskWholeConnection.h"

Connection::Connection(Component *c1, Component *c2)
{
    this->c1 = c1;
    this->c2 = c2;
}

Connection::~Connection()
{
}

void Connection::maybeMoveToGPU()
{
}

double *Connection::getEnergy()
{
    return NULL;
}

void Connection::getParallelTasks(vector<Task*> &tasks)
{
    // by default, no parallel tasks
}

void Connection::getSerialTasks(vector<Task*> &tasks)
{
    // by default, one serial task that runs the whole Connection
    tasks.push_back(new TaskWholeConnection(this));
}

