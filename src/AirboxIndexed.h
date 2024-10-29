/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * An airbox that uses an index array to allow for non-uniformities
 */
#ifndef _AIRBOX_INDEXED_H_
#define _AIRBOX_INDEXED_H_

#include "Airbox.h"

#ifdef USE_GPU
#include "GPUAirboxIndexed.h"
#endif

class AirboxIndexed : public Airbox {
 public:
    AirboxIndexed(string name, double lx, double ly, double lz, double c_a, double rho_a);
    virtual ~AirboxIndexed();

    virtual void runTimestep(int n);

    void addDrumShell(int startz, int endz, double R);
    void addDrumShell(double cx, double cy, double bz, double R, double H_shell);

    virtual int getGPUMemRequired();

    virtual void runPartialUpdate(int start, int len);

    // skip the halo when returning state pointers!
    virtual double *getU() { return &u[nxny]; }
    virtual double *getU1() { return &u1[nxny]; }
    virtual double *getU2() { return &u2[nxny]; }
    
    virtual bool isOnGPU();
    virtual bool moveToGPU();

    virtual double *getEnergy();

    virtual void addPlate(int zb, double *true_Psi);

    virtual void swapBuffers(int n);

 protected:
    AirboxIndexed(string name);

    double coeff0, coeff1, coeff2;
    double APSI_CORNER, APSI_EDGE, APSI_FACE;
    double CPSI_CORNER, CPSI_EDGE, CPSI_FACE;

    double *coeffs;
    unsigned char *index;

    virtual void generateDrumShellCoefficients();
    void addDrumShellInternal(double cx, double cy, int startz, int endz, double R);
    int nextCoeff;
    int drumShellCoeffs;    

    double th;

    double *energy;
    unsigned char *energyMask;

    bool reflection;

#ifdef USE_GPU
    GPUAirboxIndexed *gpuAirboxIndexed;
#endif
};

#endif
