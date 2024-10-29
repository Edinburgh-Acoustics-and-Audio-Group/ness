/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Represents a worker thread for multicore parallelisation
 */
#ifndef _WORKER_THREAD_H_
#define _WORKER_THREAD_H_

#include "Task.h"

#ifndef WIN32
#include <pthread.h>
#else
#include <windows.h>
#endif

#include <vector>
using namespace std;

class WorkerThread {
 public:
    // create a worker thread with the specified ID
    WorkerThread(int threadId);
    virtual ~WorkerThread();

    // get number of CPU cores in the machine
    static int getNumCores();

    // create the worker threads at startup
    static vector<WorkerThread*> *createWorkerThreads(vector <Task*> &tasks,
						      vector <Task*> &connectionTasks);

    // this is public and static because the main thread
    // will also call it to run its component(s)
    static void workerThreadRun(int threadId, int n);
    static void workerThreadConnectionRun(int threadId, int n);

    static void setGo();
    static void connectionsGo();
    static void clearGo();

    // pthread function
    static void *pthreadFunction(void *dat);

 protected:
    // ID of this thread
    int threadId;
    static int numThreads;
    static volatile int go;
    static volatile int nextTask;
    static volatile int completedThreads;
    static volatile int loopedThreads;

    static vector<Task*> *tasks;
    static vector<Task*> *connectionTasks;

#ifndef WIN32
    // pthread handle for this thread
    pthread_t pthread;
#else
    // Windows handle for this thread
    HANDLE wthread;
#endif
};

#endif
