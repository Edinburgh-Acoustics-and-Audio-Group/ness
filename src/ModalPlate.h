/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Non-linear modal plate component
 */
#ifndef _MODALPLATE_H_
#define _MODALPLATE_H_

#include "Component.h"

#ifdef USE_GPU
#include "GPUModalPlate.h"
#endif

// for optimised AVX version
struct RowInfo {
    int start; // starting index of rows of this type (in permuted H1 and t1)
    int count; // number of rows of this type
    int *q1index; // index of non-zeroes in q1 for this row type
    int nnz; // number of non-zeroes in this row type
};

// don't inherit from Component2D because the modal plate works differently from
// our standard finite difference components in many respects
class ModalPlate : public Component {
 public:
    ModalPlate(string name, int A, double Lx, double Ly, double h,
	       double nu, double Young, double rho, double loss1,
	       double loss2, bool halfNyquist = false);
    virtual ~ModalPlate();

    virtual int getIndexf(double x, double y=0.0, double z=0.0);
    virtual int getIndex(int x, int y=0, int z=0);
    virtual void getInterpolationInfo(InterpolationInfo *info, double x, double y=0.0, double z=0.0);

    virtual void runTimestep(int n);

    double getLx() { return Lx; }
    double getLy() { return Ly; }
    double *getOmega() { return ov; }
    double getH() { return h; }
    double getRho() { return rho; }
    double *getCnorm() { return Cnorm; }

    virtual int getGPUScore();
    virtual int getGPUMemRequired();
    virtual bool isOnGPU() {
#ifdef USE_GPU
	if (gpuModalPlate != NULL) return true;
#endif
	return false;
    }

    virtual bool moveToGPU();

    virtual void getParallelTasks(vector<Task*> &tasks);
    virtual void getSerialTasks(vector<Task*> &tasks);

    void runRowType(int n, int t);
    void finishUpdate(int n);

 protected:
    void instr_def(double fsd);
    double *omega_calculator();

    void optimiseForAVX();

    int eigencalcUnstable(int A, double Lx, double Ly, double *coeff1);
    double int1(int m, int p, double L);
    double int2(int m, int p, double L);
    double int4(int m, int p, double L);
    double *int2_mat(int tt, double L);

    void Hcalc_unstable(double *coeff1, int DIM, int A, double Lx, double Ly,
			int S, double *H1);
    void i_phi(int DIM, double Lx, double Ly, double *y);
    double i1_mat(int A, int DIM, double L, int m, int n, int p);
    double i2_mat(int A, int DIM, double L, int m, int n, int p);
    double i3_mat(int A, int DIM, double L, int m, int n, int p);
    double i4_mat(int A, int DIM, double L, int m, int n, int p);
    double i5_mat(int A, int DIM, double L, int m, int n, int p);
    double i9_mat(int A, int DIM, double L, int m, int n, int p);
    double i10_mat(int A, int DIM, double L, int m, int n, int p);
    double i11_mat(int A, int DIM, double L, int m, int n, int p);
    double i12_mat(int A, int DIM, double L, int m, int n, int p);
    double i13_mat(int A, int DIM, double L, int m, int n, int p);

    // dimensions in metres
    double Lx, Ly;

    int A;
    int DIM;
    double h;
    double nu;
    double Young;
    double rho;

    double loss1, loss2;

    double *ov;
    double *Cnorm;

    double *t1, *t2;
    double *H1;
    double *G;
    double *C, *C1, *C2;

    /*
     * Data for optimised AVX version
     */
    // permuted, cache-friendly version of H1
    double *H1vals;

    // index into row types for each row
    int *rowTypeIndex;

    // information for each row type
    RowInfo *rowInfo;

    // number of rows of each row type
    int *rowTypeCounts;

    // forward and reverse permutations for collecting all rows of a given type together
    int *rowPerm;
    int *reversePerm;

    // number of row types discovered so far
    int numRowTypes;

    // permuted version of t1
    double *t1perm;

#ifdef USE_AVX
    void avxUpdate4Rows(double *q1, double *H1, double *result, int *q1index,
			int nnz);
    void avxUpdateSingleRow(double *q1, double *H1, double *result, int *q1index,
			    int nnz);
#endif

#ifdef USE_GPU
    GPUModalPlate *gpuModalPlate;
#endif
};

#endif
