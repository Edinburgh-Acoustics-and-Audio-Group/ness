/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "Bar.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"

#include <cmath>
using namespace std;

Bar::Bar(string name, Material *mat, double L, double H0, int bc)
    : Component1D(name)
{
    init(mat->getYoungsModulus(), mat->getDensity(), L, H0, bc);
}

Bar::Bar(string name, double E, double rho_0, double L, double H0, int bc)
    : Component1D(name)
{
    init(E, rho_0, L, H0, bc);
}

Bar::~Bar()
{
    CSR_free(B);
}

void Bar::runTimestep(int n)
{
    int i;
    CSR_matrix_vector_mult(B, u1, u);
    for (i = 0; i < ss; i++) {
	u[i] = u[i] - u2[i];
    }
    runInputs(n, u, u1, u2);
}

void Bar::logMatrices()
{
    saveMatrix(B, "B");
}

// FIXME: add loss

void Bar::init(double E, double rho_0, double L, double H0, int bc)
{
    this->E = E;
    this->rho_0 = rho_0;
    this->L = L;
    this->H0 = H0;
    this->bc = bc;

    // calculate scalars
    A = H0 * H0;
    I0 = (H0*H0*H0*H0) / 12.0;
    kappa = sqrt(E * I0 / (rho_0 * A));
    double k = GlobalSettings::getInstance()->getK();
    h = sqrt(2.0 * kappa * k);

    h = h * SettingsManager::getInstance()->getDoubleSetting(name, "fixpar");

    int N = (int)floor(L / h);
    h = L / (double)N;
    mu = (kappa * k) / (h * h);

    alpha = (k * k) / (rho_0 * H0 * h * h);
    bowFactor = k / (2.0 * h * h);

    allocateState(N+1);

    // compute update matrix
    int i;
    double *coeffs1, *coeffs2;
    CSRmatrix *D, *D1, *D2, *tm1, *tm2, *tm3, *eye;

    coeffs1 = new double[ss];
    coeffs2 = new double[ss];
    for (i = 0; i < ss; i++) {
	coeffs1[i] = 0.0;
	coeffs2[i] = 0.0;
    }
    coeffs1[0] = -1.0;
    coeffs2[0] = -1.0;
    coeffs2[1] = 1.0;

    D = CSR_toeplitz(coeffs1, N, coeffs2, N+1);
    delete[] coeffs1;
    delete[] coeffs2;

    D1 = CSR_transpose(D);
    CSR_scalar_computation(D1, -1.0, 3);
    if ((bc == 1) || (bc == 2)) {
	CSRSetValue(D1, 0, 0, 0.0);
	CSRSetValue(D1, N, N-1, 0.0);
    }
    else {
	CSRSetValue(D1, 0, 0, 2.0);
	CSRSetValue(D1, N, N-1, -2.0);
    }

    D2 = CSR_transpose(D);
    CSR_scalar_computation(D2, -1.0, 3);
    if ((bc == 2) || (bc == 3)) {
	CSRSetValue(D2, 0, 0, 0.0);
	CSRSetValue(D2, N, N-1, 0.0);
    }
    else {
	CSRSetValue(D2, 0, 0, 2.0);
	CSRSetValue(D2, N, N-1, -2.0);
    }

    eye = CSR_create_eye(N+1);
    CSR_scalar_computation(eye, 2.0, 3);

    tm1 = CSR_matrix_multiply(D2, D);

    tm2 = CSR_matrix_multiply(D1, D);
    CSR_free(D);
    CSR_free(D1);
    CSR_free(D2);

    tm3 = CSR_matrix_multiply(tm2, tm1);
    CSR_free(tm1);
    CSR_free(tm2);

    CSR_scalar_computation(tm3, (mu*mu), 3);

    B = CSR_matrix_add_sub(eye, tm3, 1);
    CSR_free(tm3);
    CSR_free(eye);
}
