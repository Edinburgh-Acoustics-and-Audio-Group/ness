/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Parallel task support for NESS
 */

#ifndef _TASK_H_
#define _TASK_H_

class Task {
 public:
    virtual ~Task();
    
    virtual void runTimestep(int n) = 0;

 protected:
    Task();
};

#endif
