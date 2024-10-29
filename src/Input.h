/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Represents any input to the simulation. Each input happens at a
 * defined point on a certain component.
 */
#ifndef _INPUT_H_
#define _INPUT_H_

#include "Component.h"

class Input {
 public:
    Input(Component *comp, double x, double y=0, double z=0);
    Input(Component *comp);
    virtual ~Input();
    virtual void runTimestep(int n, double *s, double *s1, double *s2) = 0;
    virtual void moveToGPU() = 0;

    static int getFirstInputTimestep() { return firstInputTimestep; }

    // allow other parts of the code to prevent early timesteps from being skipped
    static void setFirstInputTimestep(int ts) {
	if (ts < firstInputTimestep) firstInputTimestep = ts;
    }

    // most inputs prevent energy being conserved, which is a problem when this is
    // being checked. For the few that don't (e.g. brass instrument valves), this
    // should be overridden to return false
    virtual bool preventsEnergyConservation() {
	return true;
    }

 protected:
    // utility function to convert a time in seconds to a
    // timestep in samples
    int timeToTimestep(double tm);

    // component being input to
    Component *component;

    // grid point within component
    int index;

    // first timestep where this input is active
    int startTime;

    // number of timesteps during which input is active
    int duration;

    static int firstInputTimestep;
};

#endif
