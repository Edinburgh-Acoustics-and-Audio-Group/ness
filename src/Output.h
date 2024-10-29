/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Represents an output taken from one of the components in a simulation.
 * This is a usable concrete class in itself for default outputs; subclasses
 * will handle other cases like pressure outputs and interpolation.
 */
#ifndef _OUTPUT_H_
#define _OUTPUT_H_

#include "Component.h"

#ifdef USE_GPU
#include "GPUOutput.h"
#endif

#include <string>
using namespace std;

class Output {
 public:
    Output(Component *comp, double pan, double x, double y = 0.0, double z = 0.0, int interpolated = -1);
    Output();
    virtual ~Output();

    virtual void runTimestep(int n);
    
    double *getData() { return data; }
    double getPan() { return pan; }

    void saveRawData(string filename);
    void highPassFilter();
    double getMaxValue();
    void normalise();

    virtual void maybeMoveToGPU();
    void copyFromGPU();

    Component *getComponent() { return component; }
    
 protected:
    // output data at each timestep
    double *data;

    // position within stereo mix (-1 to 1)
    double pan;

    // component output is taken from
    Component *component;

    // grid point within component
    int index;

    InterpolationInfo interp;

#ifdef USE_GPU
    GPUOutput *gpuOutput;
#endif
};

#endif
