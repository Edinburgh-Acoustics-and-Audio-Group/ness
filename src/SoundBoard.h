/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * A 2D soundboard with strings attached
 */
#ifndef _SOUNDBOARD_H_
#define _SOUNDBOARD_H_

#include "Plate.h"
#include "ComponentString.h"

#include <vector>
using namespace std;

class SoundBoard : public Plate {
 public:
    SoundBoard(string name, double nu, double rho, double E, double thickness, double tension,
	       double lx, double ly, double t60_0, double t60_1000, int bc,
	       vector<ComponentString*> *strings);
    virtual ~SoundBoard();

    virtual void runTimestep(int n);
    virtual void logMatrices();

    virtual int getGPUScore();

    virtual bool isThreadable();

 protected:
    // connection matrices
    CSRmatrix *IpB, *IpC, *Jp;
    CSRmatrix *Mspinv;

    // vectors used in connections
    double *l, *fsp;

    vector<ComponentString*> *strings;
};

#endif
