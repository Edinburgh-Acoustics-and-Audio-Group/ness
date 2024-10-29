/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */
#include "WorkerThread.h"

#include "GlobalSettings.h"
#include "Logger.h"
#include "Input.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
using namespace std;

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef WIN32
#include <windows.h>
#endif

int WorkerThread::numThreads = 0;
volatile int WorkerThread::go = 0;
volatile int WorkerThread::nextTask = 0;
volatile int WorkerThread::completedThreads = 0;
volatile int WorkerThread::loopedThreads = 0;

vector<Task*> *WorkerThread::tasks = NULL;
vector<Task*> *WorkerThread::connectionTasks = NULL;

#ifdef WIN32
DWORD WINAPI NESSWindowsThreadProc(LPVOID param)
{
    // call the Pthreads function
    WorkerThread::pthreadFunction(param);
    return 0;
}
#endif


WorkerThread::WorkerThread(int threadId)
{
    this->threadId = threadId;

#ifndef WIN32
    // create actual Pthread
    if (pthread_create(&pthread, NULL, pthreadFunction, (void *)this) != 0) {
	logMessage(5, "Error creating worker thread");
	exit(1);
    }
#else
    wthread = CreateThread(NULL, 0, NESSWindowsThreadProc, (LPVOID)this, 0, NULL);
    if (wthread == NULL) {
	logMessage(5, "Error creating worker thread");
	exit(1);
    }
#endif
}


WorkerThread::~WorkerThread()
{
}


int WorkerThread::getNumCores()
{
#ifndef WIN32
#ifdef __APPLE__
    int ncore;
    size_t count_len = sizeof(ncore);
    sysctlbyname("hw.logicalcpu", &ncore, &count_len, NULL, 0);
    return ncore;
#else
    int ncore = 0;
    char linebuf[1000];

    // count "processor" lines in cpuinfo
    ifstream in("/proc/cpuinfo");

    if (in.fail()) return 1;

    while (!in.eof()) {
	in.getline(linebuf, 1000);

	if (!strncmp(linebuf, "processor", 9)) {
	    ncore++;
	}
    }

    in.close();

    return ncore;
#endif
#else
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;    
#endif
}

vector<WorkerThread*> *WorkerThread::createWorkerThreads(vector<Task*> &tasks,
							 vector<Task*> &connectionTasks)
{
    int i;
    vector<WorkerThread*> *threads = new vector<WorkerThread*>();
    int numCores = getNumCores();

    //components = comps;
    WorkerThread::tasks = &tasks;
    WorkerThread::connectionTasks = &connectionTasks;
    numThreads = numCores;

    // if number of tasks is less than number of cores, only create one thread
    // per task
    if (tasks.size() < numThreads) numThreads = tasks.size();

    // override number of threads created
    int maxThreads = GlobalSettings::getInstance()->getMaxThreads();
    if (numThreads > maxThreads) numThreads = maxThreads;

    // create one for each core, minus the main thread
    for (i = 1; i < numThreads; i++) {
	WorkerThread *wt = new WorkerThread(i);
	threads->push_back(wt);
    }

    logMessage(3, "Number of cores: %d, number of tasks: %d, number of threads: %d",
	       numCores, tasks.size(), numThreads);
    return threads;
}

// performs an atomic increment of the volatile variable, and returns its previous value
static int atomicIncrement(volatile int* val)
{
#ifndef WIN32
    return __sync_fetch_and_add(val, 1);
#else
    return InterlockedIncrement((LONG*)val) - 1;
#endif
}

void WorkerThread::workerThreadConnectionRun(int threadId, int n)
{
    int next;

    // run our "own" task first, if we have one
    if (threadId < connectionTasks->size()) {
	connectionTasks->at(threadId)->runTimestep(n);

	// fetch index of next task needing run
    next = atomicIncrement(&nextTask);
	while (next < connectionTasks->size()) {
	    // run it
	    connectionTasks->at(next)->runTimestep(n);
        next = atomicIncrement(&nextTask);
	}
    }

    // signal that we've finished
    atomicIncrement(&completedThreads);
}

void WorkerThread::workerThreadRun(int threadId, int n)
{
    int next;

    // run our "own" task first
    tasks->at(threadId)->runTimestep(n);

    // fetch index of next task needing run
    next = atomicIncrement(&nextTask);
    while (next < tasks->size()) {
	// run it
	tasks->at(next)->runTimestep(n);
    next = atomicIncrement(&nextTask);
    }

    // signal that we've finished
    atomicIncrement(&completedThreads);

    // spin waiting for all threads to complete
    //while (completedThreads < numThreads);
}

void WorkerThread::setGo()
{
    completedThreads = 0;
    nextTask = numThreads;
    go = 1;
}

void WorkerThread::connectionsGo()
{
    completedThreads = 0;
    nextTask = numThreads;
    go = 1;
}

void WorkerThread::clearGo()
{
    while (completedThreads < numThreads);
    loopedThreads = 1;
    go = 0;

    // make sure all the threads have cleared this barrier or one could get deadlocked
    while (loopedThreads < numThreads);
}

void *WorkerThread::pthreadFunction(void *dat)
{
    WorkerThread *thread = (WorkerThread *)dat;
    int nn;
    int start;

    int NF = GlobalSettings::getInstance()->getNumTimesteps();

    logMessage(1, "Worker thread %d starting up", thread->threadId);

    if (GlobalSettings::getInstance()->getEstimate()) {
	if (NF > 1000) {
	    NF = 1000;
	}
	start = 0;
    }
    else {
	start = Input::getFirstInputTimestep();
    }

    // loop over simulation timesteps
    for (nn = start; nn < NF; nn++) {

	// wait til it's time to run components
	while (!go);

	// run them
	workerThreadRun(thread->threadId, nn);

	// wait for "go" flag to be cleared
	while (go);

	// let the main thread know we're through the barrier
    atomicIncrement(&loopedThreads);

	if (connectionTasks->size() > 0) {
	    // wait til it's time to run connections
	    while (!go);

	    // run them
	    workerThreadConnectionRun(thread->threadId, nn);

	    // wait for "go" flag to be cleared
	    while (go);

	    // let the main thread know we're through the barrier
        atomicIncrement(&loopedThreads);
	}
    }

    logMessage(1, "Worker thread %d terminating", thread->threadId);

    // done
    return NULL;
}

