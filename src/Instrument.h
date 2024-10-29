/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * This class represents an instrument, which is a complete set of components
 * and other entities used to run a simulation.
 */
#ifndef _INSTRUMENT_H_
#define _INSTRUMENT_H_

#include "Component.h"
#include "Output.h"
#include "Connection.h"
#include "Profiler.h"
#include "Task.h"

#ifndef NO_THREADS
#include "WorkerThread.h"
#endif

#include <vector>
#include <string>
using namespace std;

class Instrument {
 public:
    Instrument();
    virtual ~Instrument();

    void addComponent(Component *comp) { components.push_back(comp); }
    void addOutput(Output *output) { outputs.push_back(output); }
    void addConnection(Connection *conn) { connections.push_back(conn); }

    virtual void runTimestep(int n);
    virtual void endTimestep(int n);

    void saveOutputs(string outputname, bool individual, bool raw);

    Component *getComponentByName(string name);

    void optimise();

    vector<Component*> *getComponents() { return &components; }
    vector<Output*> *getOutputs() { return &outputs; }
    vector<Connection*> *getConnections() { return &connections; }

 protected:
    double getMaxOutput();

    // all the components
    vector<Component*> components;
    vector<Output*> outputs;
    vector<Connection*> connections;

    // tasks that can't be run concurrently with other tasks
    vector<Task*> serialTasks;

    // tasks that are run on multiple threads on the CPU
    vector<Task*> parallelTasks;

    // components that are run on the GPU
    vector<Component*> gpuComponents;

    // connection tasks that have to run in serial
    vector<Task*> serialConnectionTasks;

    // connection tasks that can run in parallel
    vector<Task*> parallelConnectionTasks;

#ifndef NO_THREADS
    vector<WorkerThread*> *workerThreads;
#endif

    Profiler *profiler;
};

#endif
