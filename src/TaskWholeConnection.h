/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * A parallel task that performs the entire update for one connection
 */

#ifndef _TASKWHOLECONNECTION_H_
#define _TASKWHOLECONNECTION_H_

#include "Task.h"
#include "Connection.h"

class TaskWholeConnection : public Task {
 public:
    TaskWholeConnection(Connection *conn);
    virtual ~TaskWholeConnection();

    virtual void runTimestep(int n);

 protected:
    Connection *connection;
};

#endif
