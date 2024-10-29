/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * String-to-string connection as used in net1 code
 */

#ifndef _CONNECTIONNET1_H_
#define _CONNECTIONNET1_H_

#include "Connection.h"

#include <cstdio>

class ConnectionNet1 : public Connection {
 public:
    ConnectionNet1(Component *c1, Component *c2, double mass, double freq,
		   double loss, double collisionExponent, double rattlingDistance,
		   double x1, double x2);
    virtual ~ConnectionNet1();

    virtual void runTimestep(int n);
    virtual double *getEnergy();

    virtual void getParallelTasks(vector<Task*> &tasks);
    virtual void getSerialTasks(vector<Task*> &tasks);

    bool coincides(ConnectionNet1 *conn);

    void setIterations(int in) {
	itnum = in;
    }

 protected:
    double mass;
    double freq;
    double loss;
    double collisionExponent;
    double rattlingDistance;
    double x1;
    double x2;

    double Gs1, Gs2, Gc;
    double V;
    double Q;
    double Wc[4];
    double Mc[4];

    double u, u1, u2;

    int itnum;

    int i1, i2;

    double *energy;
};

#endif
