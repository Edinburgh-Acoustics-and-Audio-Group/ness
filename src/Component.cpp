/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */
#include "Component.h"
#include "Input.h"
#include "SettingsManager.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "TaskWholeComponent.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
using namespace std;

Component::Component(string name)
{
    this->name = name;
    parent = NULL;
    u = NULL;
    u1 = NULL;
    u2 = NULL;

    stateStream = NULL;
    logState = SettingsManager::getInstance()->getBoolSetting(name, "log_state");
}

Component::~Component()
{
    int i;
    if (u) delete[] u;
    if (u1) delete[] u1;
    if (u2) delete[] u2;
    for (i = 0; i < inputs.size(); i++) {
	delete inputs[i];
    }

    if (stateStream != NULL) {
	stateStream->close();
	delete stateStream;
    }
}

void Component::doSaveState()
{
    if (logState) {
	if (stateStream == NULL) {
	    char filenamebuf[1000];
	    sprintf(filenamebuf, "%s-state.bin", name.c_str());
	    stateStream = new ofstream(filenamebuf, ios::out | ios::binary);
	}
	stateStream->write((const char *)u, ss * sizeof(double));
    }
}

void Component::swapBuffers(int n)
{
    double *tmp;

    doSaveState();

    tmp = u2;
    u2 = u1;
    u1 = u;
    u = tmp;
}

void Component::runInputs(int n, double *s, double *s1, double *s2)
{
    int i;
    bool energyOn = GlobalSettings::getInstance()->getEnergyOn();
    for (i = 0; i < inputs.size(); i++) {
	// if we're checking energy conservation, don't run the inputs that mess that up
	if ((!energyOn) || (!inputs[i]->preventsEnergyConservation())) {
	    inputs[i]->runTimestep(n, s, s1, s2);
	}
    }
}

#ifndef BRASS_ONLY

void Component::saveMatrix(CSRmatrix *mat, string name, int idx)
{
    char filenamebuf[1000];
    char filenamebuf2[1000];
    if (idx < 0) {
	sprintf(filenamebuf, "%s-%s.txt", this->name.c_str(), name.c_str());
	sprintf(filenamebuf2, "%s-%s.mat", this->name.c_str(), name.c_str());
    }
    else {
	sprintf(filenamebuf, "%s-%s-%d.txt", this->name.c_str(), name.c_str(), idx);
	sprintf(filenamebuf2, "%s-%s-%d.mat", this->name.c_str(), name.c_str(), idx);
    }

    // save text version (Matlab format)
    CSRMatlabPrint(mat, filenamebuf);

    // save binary version (PETSc format)
    CSR_save_petsc(filenamebuf2, mat);
}

#ifndef GUITAR_ONLY

void Component::saveMatrix5x5(matrix_5x5_t *mat, string name, int idx)
{
    CSRmatrix *tmp = m5x5ToCSR(mat);
    saveMatrix(tmp, name, idx);
    CSR_free(tmp);
}

void Component::saveMatrix3x3(matrix_3x3_t *mat, string name, int idx)
{
    CSRmatrix *tmp = m3x3ToCSR(mat);
    saveMatrix(tmp, name, idx);
    CSR_free(tmp);
}

#endif

void Component::saveVector(double *vec, int len, string name, int idx)
{
    char filenamebuf[1000];
    if (idx < 0) {
	sprintf(filenamebuf, "%s-%s.bin", this->name.c_str(), name.c_str());
    }
    else {
	sprintf(filenamebuf, "%s-%s-%d.bin", this->name.c_str(), name.c_str(), idx);
    }

    ofstream of(filenamebuf, ios::out | ios::binary);
    if (!of.good()) return;
    of.write((const char *)vec, len * sizeof(double));
    of.close();
}

int Component::getMatrixMemRequired(CSRmatrix *mat)
{
    return ((mat->nrow + 1) * sizeof(int)) + // row starts
	(mat->rowStart[mat->nrow] * sizeof(int)) + // column index
	(mat->rowStart[mat->nrow] * sizeof(double)); // values
}

#endif

void Component::logMatrices()
{
}

int Component::getGPUScore()
{
    return GPU_SCORE_NO;
}

int Component::getGPUMemRequired()
{
    // by default, just the state arrays
    return ss * 3 * sizeof(double);
}

bool Component::isThreadable()
{
    // most components are. only ones that access another component's
    // state array in their runTimestep are not.
    return true;
}

void Component::getParallelTasks(vector<Task*> &tasks)
{
    // by default a single parallel task handles the whole component, if it's a
    // parallel component
    if (isThreadable()) {
	tasks.push_back(new TaskWholeComponent(this));
    }
}

void Component::getSerialTasks(vector<Task*> &tasks)
{
    // by default, no serial tasks, unless it's not threadable in which case whole
    // component is in a single serial task
    if (!isThreadable()) {
	tasks.push_back(new TaskWholeComponent(this));
    }
}

bool Component::moveToGPU()
{
    return false;
}

bool Component::isOnGPU()
{
    return false;
}

double *Component::getEnergy()
{
    return NULL;
}

// ss is passed as a parameter here because it may include a halo or something
// that's not included in the member variable ss!
void Component::initialiseState(int ss)
{
    int i;

    memset(u, 0, ss * sizeof(double));
    memset(u1, 0, ss * sizeof(double));
    memset(u2, 0, ss * sizeof(double));

    SettingsManager *sm = SettingsManager::getInstance();
    if (sm->getBoolSetting(name, "impulse")) {
	// add impulse at centre of domain
	u1[getIndexf(0.5, 0.5, 0.5)] = 0.001;
	logMessage(1, "Impulse going in at %d", getIndexf(0.5, 0.5, 0.5));
    }
}
