/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Represents a 2D plate - e.g. a metal plate as used in the Zero code
 */
#ifndef _PLATE_H_
#define _PLATE_H_

extern "C" {
#include "csrmatrix.h"  
};

#include "Component2D.h"
#include "Material.h"

#ifdef USE_GPU
#include "GPUPlate.h"
#endif

class Plate : public Component2D {
 public:
    Plate(string name, Material *material, double thickness, double tension,
	  double lx, double ly, double t60_0, double t60_1000, int bc);
    Plate(string name, double nu, double rho, double E, double thickness, double tension,
	  double lx, double ly, double t60_0, double t60_1000, int bc);
    virtual ~Plate();

    virtual void runTimestep(int n);
    virtual void swapBuffers(int n);

    virtual void logMatrices();

    virtual int getGPUScore();
    virtual int getGPUMemRequired();

    virtual bool isOnGPU() {
#ifdef USE_GPU
	if (gpuPlate != NULL) return true;
#endif
	return false;	
    }

    virtual bool moveToGPU();

    // skip the halo when returning state pointers, when on GPU!
    virtual double *getU() {
	if (isOnGPU()) return &u[ny+ny];
	return u;
    }
    virtual double *getU1() {
	if (isOnGPU()) return &u1[ny+ny];
	return u1;
    }
    virtual double *getU2() {
	if (isOnGPU()) return &u2[ny+ny];
	return u2;
    }

    virtual double *getEnergy();

 protected:
    void init(double nu, double rho, double E, double thickness, double tension,
	      double lx, double ly, double t60_0, double t60_1000, int bc);

    double d0;
    double nu0;
    double rho0;
    double h0;
    double t0;
    double kappa;
    double c;
    double sig_0;
    double sig_1;
    double h;
    double mu;
    double lambda;

    CSRmatrix *B, *C;

    // energy check stuff
    double *energy;
    CSRmatrix *D2_mat, *D3_mat, *Dxy2_mat, *Dx_mat, *Dy_mat;
    double *scalevec_t, *scalevec_mxmy, *scalevec_mxmy2;
    double *scalevec_mxy, *scalevec_x_total, *scalevec_y_total;
    double *mx, *mx1, *my, *my1, *mxy, *mxy1;
    double *Dxu, *Dxu1, *Dyu, *Dyu1;

#ifdef USE_GPU
    GPUPlate *gpuPlate;
#endif
};


#endif
