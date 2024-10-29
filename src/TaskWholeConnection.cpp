/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TaskWholeConnection.h"

TaskWholeConnection::TaskWholeConnection(Connection *conn)
{
    connection = conn;
}

TaskWholeConnection::~TaskWholeConnection()
{
}

void TaskWholeConnection::runTimestep(int n)
{
    connection->runTimestep(n);
}

