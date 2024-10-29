/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * This class represents any component within a simulation, with subclasses
 * for 1D, 2D and 3D components and further subclasses for specific types
 * (e.g. plate, board, bar, string, airbox). It assumes that every
 * component has 3 state arrays containing its state at the most recent
 * 3 timesteps. Any component that needs more than this will have to manage
 * them itself in a subclass.
 */
#ifndef _COMPONENT_H_
#define _COMPONENT_H_

#ifndef BRASS_ONLY
extern "C" {
#include "csrmatrix.h"
#ifndef GUITAR_ONLY
#include "banded.h"
#endif
};
#endif

#include "Task.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

class Input;

enum { GPU_SCORE_NO, GPU_SCORE_GOOD, GPU_SCORE_GREAT };

/*
 * Information for linear interpolated input or output
 */
enum { INTERPOLATION_NONE, INTERPOLATION_LINEAR, INTERPOLATION_BILINEAR,
       INTERPOLATION_TRILINEAR };
struct InterpolationInfo {
    int type;          // interpolation type (see enum above)
    int idx;           // index of base point
    int nx;            // row stride for 2D or 3D component
    int nxny;          // layer stride for 3D component
    double alpha[8];   // alpha values for each point
};

class Component {
 public:
    Component(string name);
    virtual ~Component();

    // getters for state arrays and overall state size
    virtual double *getU() { return u; }
    virtual double *getU1() { return u1; }
    virtual double *getU2() { return u2; }
    int getStateSize() { return ss; }

    string getName() { return name; }

    double getAlpha() { return alpha; }
    double getBowFactor() { return bowFactor; }

    // component hierarchy
    Component *getParent() { return parent; }
    vector<Component*> *getChildren() { return &children; }

    void addInput(Input *input) { inputs.push_back(input); }
    vector<Input*> *getInputs() { return &inputs; }

    // gets the index of a point within the component's state arrays
    // getIndexf takes values normalised to the range 0-1
    // getIndex takes co-ordinates in the component grid
    virtual int getIndexf(double x, double y=0.0, double z=0.0) = 0;
    virtual int getIndex(int x, int y=0, int z=0) = 0;

    virtual void getInterpolationInfo(InterpolationInfo *info, double x, double y=0.0, double z=0.0) = 0;

    // swap the component's state buffers, used at end of timestep
    virtual void swapBuffers(int n);

    // run one timestep (abstract, specific to component type)
    virtual void runTimestep(int n) = 0;

    void setLogState(bool ls) { logState = ls; }

    virtual void logMatrices();

    virtual int getGPUScore();
    virtual int getGPUMemRequired();
    virtual bool isThreadable();

    virtual bool isOnGPU();
    virtual bool moveToGPU();

    virtual double *getEnergy();

    virtual void getParallelTasks(vector<Task*> &tasks);
    virtual void getSerialTasks(vector<Task*> &tasks);

 protected:
    void runInputs(int n, double *s, double *s1, double *s2);

#ifndef BRASS_ONLY
    void saveMatrix(CSRmatrix *mat, string name, int idx = -1);
    void saveVector(double *vec, int len, string name, int idx = -1);
#ifndef GUITAR_ONLY
    void saveMatrix5x5(matrix_5x5_t *mat, string name, int idx = -1);
    void saveMatrix3x3(matrix_3x3_t *mat, string name, int idx = -1);
#endif

    int getMatrixMemRequired(CSRmatrix *mat);
#endif

    void doSaveState();

    void initialiseState(int ss);

    // name
    string name;

    // state arrays
    double *u, *u1, *u2;

    // overall state size
    int ss;

    double alpha;
    double bowFactor;

    // hierarchical relationships
    Component *parent;
    vector<Component*> children;

    // inputs targetting this component
    vector<Input*> inputs;

    // save out state every timestep!
    bool logState;

    // file for logging state to
    ofstream *stateStream;
};

#endif
