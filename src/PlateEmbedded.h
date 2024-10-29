/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * A plate embedded within an airbox.
 */
#ifndef _PLATEEMBEDDED_H_
#define _PLATEEMBEDDED_H_

#include "Component2D.h"
#include "Material.h"
#include "Airbox.h"
#include "Profiler.h"

extern "C" {
#include "csrmatrix.h"
#include "pcg.h"
#include "banded.h"
};

class PlateEmbedded : public Component2D {
 public:
    PlateEmbedded(string name, Material *material, double thickness, double tension,
		  double lx, double ly, double t60_0, double sig1, double cx, double cy, double cz,
		  bool isMembrane = false, bool isCircular = false);
    PlateEmbedded(string name, double nu, double rho, double E, double thickness,
		  double tension, double lx, double ly, double t60_0, double sig1, double cx, double cy,
		  double cz, bool isMembrane = false, bool isCircular = false);
    virtual ~PlateEmbedded();

    virtual void runTimestep(int n);

    virtual void swapBuffers(int n);

    virtual void logMatrices();

    virtual int getGPUScore();

    double getCx() { return cx; }
    double getCy() { return cy; }
    double getCz() { return cz; }

    double getLx() { return lx; }
    double getLy() { return ly; }

    double getK() { return k; }
    double getA0() { return a0; }
    double getRho() { return rho; }
    double getThickness() { return H; }
    double getH() { return h; }
    
    int *getTruePhi() { return TruePhi; }
    int getTruePhiSize() { return TruePhiSize; }

    void setInterpolation(CSRmatrix *IMat, CSRmatrix *JMat, double *buffer, double cplfac);

    bool getCircular() { return isCircular; }
    bool getMembrane() { return isMembrane; }

    virtual double *getEnergy();

 protected:
    // initialisation functions
    void init(double nu, double rho, double E, double thickness,
	      double tension, double lx, double ly, double t60_0, double sig1,
	      double cx, double cy, double cz, bool isMembrane, bool isCircular);
    int myBiharmRect(double nu, int Sx, int Sy, int BC_flag, int BCVersion,
		     CSRmatrix **DDDD, CSRmatrix **DD, CSRmatrix **Dxx_2, CSRmatrix **Dyy_2,
		     CSRmatrix **Dxy_2);
    int myAiryRect(double nu, int Sx, int Sy, int BC_flag, int BCVersion,
		   CSRmatrix **DDDD_F, CSRmatrix **DD_F, CSRmatrix **Dxx_2F,
		   CSRmatrix **Dyy_2F);
    int newAiryCircular(int S, CSRmatrix **DDDD_F, CSRmatrix **DD_F);
    int preAllocateMatrices(CSRmatrix *DLMM, CSRmatrix *DLMP, CSRmatrix *DLPM,
			    CSRmatrix *DLPP, CSRmatrix *Dxxw, CSRmatrix *Dyyw,
			    CSRmatrix *Dxx_F, CSRmatrix *Dyy_F);
    void generateDifferenceMatrices(CSRmatrix *DDDD, CSRmatrix *Dxxw,
				    CSRmatrix *Dyyw, double K, double a0,
				    double k, double c);
    void generateDLMatrices(CSRmatrix **DLMM, CSRmatrix **DLMP, CSRmatrix **DLPM,
			    CSRmatrix **DLPP, int Nx, int Ny);


    // timestep functions
    void applyIMat(double *source, double *dest, double scalar);
    void computeTemp1_6();
    void computeLOPWF();
    void computeBigMat();
    void computeSYMASY();


    // Phi state
    double *Phi, *Phi1, *Phi0;

    // physical size in metres
    double lx, ly;

    // centre position
    double cx, cy, cz;

    // scalars
    double rho, H, E, nu, T60, sig1, T, D;
    double h, k, a0;

    double CF;
    double BL;
    double cplfac;

    // list of actual active elements
    int TruePhiSize;
    int *TruePhi;

    // band size in the banded matrices
    int bandSize;

    // true if a drum membrane rather than a rigid plate
    bool isMembrane;

    // true if circular rather than rectangular
    bool isCircular;

    CSRmatrix *DDDD_F;
    CSRmatrix *LOPW, *LOPF;
    CSRmatrix *BigMat;

    // difference matrices
    CSRmatrix *B, *C, *Bw;

    // interpolation matrices
    CSRmatrix *IMat, *JMat;

    double *interpBuffer;

    // temporary vectors
    double *temp1, *temp2, *temp3, *temp4, *temp5, *temp6;
    double *vtemp;
    double *aa, *bb;
    double *Right, *BigRight;

    double *Diff_tmp1;

    bool linear;
    bool symmetric;

    int iterinv;

    // solver
    pcg_info_t *pcg;

    /* banded versions of some of the matrices */
    matrix_5x5_t *BigMat5x5;
    matrix_3x3_t *LOPW3x3, *LOPF3x3, *cplfacIMatJMat3x3;

    matrix_3x3_t *DLMM3, *DLMP3, *DLPM3, *DLPP3;
    matrix_3x3_t *Dxxw3, *Dyyw3, *Dxx_F3, *Dyy_F3;

    matrix_5x5_t *SYM, *ASY;

    double dlmmval;

    /* energy stuff */
    double *energy;
    CSRmatrix *DD, *DD_F;
    CSRmatrix *Dxp, *Dyp;
    double *energytmp, *energytmp2;

    Profiler *profiler;
};

#endif
