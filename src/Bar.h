/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Square metal bar
 */

#ifndef _BAR_H_
#define _BAR_H_

#include "Component1D.h"
#include "Material.h"

extern "C" {
#include "csrmatrix.h"
};

class Bar : public Component1D {
 public:
    Bar(string name, Material *mat, double L, double H0, int bc);
    Bar(string name, double E, double rho_0, double L, double H0, int bc);
    virtual ~Bar();

    virtual void runTimestep(int n);
    virtual void logMatrices();

 protected:
    void init(double E, double rho_0, double L, double H0, int bc);

    double E;     // Young's modulus
    double rho_0; // density
    double L;     // length
    double H0;    // width
    int bc;       // boundary condition - 1=clamped, 2=simply supported, 3=free

    double A;     // area
    double I0;    // moment of inertia
    double kappa; // stiffness constant
    double h;     // grid spacing
    double mu;    // Courant number

    CSRmatrix *B; // update matrix
};

#endif
