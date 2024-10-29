/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "ComponentString.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"
#include "MathUtil.h"

#include <cstdio>
#include <cmath>
using namespace std;

ComponentString::ComponentString(string name, double L, double rho, double T, double E,
				 double r, double T60_0, double T60_1000, double xc1,
				 double yc1, double xc2, double yc2)
    : Component1D(name)
{
    int i;
    this->L = L;
    this->rho = rho;
    this->T = T;
    this->E = E;
    this->radius = r;
    this->T60_0 = T60_0;
    this->T60_1000 = T60_1000;

    this->xc1 = xc1;
    this->yc1 = yc1;
    this->xc2 = xc2;
    this->yc2 = yc2;

    // calculate scalars
    double sr = GlobalSettings::getInstance()->getSampleRate();
    k = 1.0 / sr;
    double A = M_PI * r * r;
    double I = 0.25 * M_PI * r * r * r * r;
    double c = sqrt(T / rho);
    double kappa = sqrt(E * I / rho);
    sig0 = 6.0 * log(10) / T60_0;
    double z1 = (-(c*c) + sqrt(c*c*c*c + 4.0*kappa*kappa * (2.0*M_PI*1000.0)*(2.0*M_PI*1000.0))) /
	(2.0*kappa*kappa);
    if (kappa == 0.0) z1 = ((2.0*M_PI*1000.0)*(2.0*M_PI*1000.0)) / (c*c);
    if (c == 0.0) z1 = (2.0*M_PI*1000.0) / kappa;
    double sig1 = (6.0*log(10) / z1) * ((1.0 / T60_1000) - (1.0 / T60_0));

    SettingsManager *sm = SettingsManager::getInstance();
    switch (sm->getIntSetting(name, "loss_mode")) {
    case 0:
	sig0 = 0.0;
	sig1 = 0.0;
	break;
    case -1:
	sig0 = 0.0;
	break;
    }

    h = sqrt(0.5 * (c*c*k*k + 4.0*sig1*k + sqrt((c*c*k*k+4.0*sig1*k)*(c*c*k*k+4.0*sig1*k) + 16.0*kappa*kappa*k*k)));
    h = h * sm->getDoubleSetting(name, "fixpar");

    int N = (int)floor(L / h);
    h = L / ((double)N);
    
    logMessage(1, "k=%f, A=%f, I=%f, c=%f, kappa=%f, sig0=%f, z1=%f, sig1=%f, N=%d, h=%f",
	       k, A, I, c, kappa, sig0, z1, sig1, N, h);

    allocateState(N+1);

    // create update matrices
    double *diag1 = new double[N+1];
    for (i = 0; i < (N+1); i++) {
	diag1[i] = 0.0;
    }
    diag1[0] = -2.0 / (h*h);
    diag1[1] = 1.0 / (h*h);

    CSRmatrix *Dxx = CSR_sym_toeplitz(diag1, N+1);
    CSRSetValue(Dxx, 0, 0, (-1.0 / (h*h)));
    CSRSetValue(Dxx, 0, 1, (1.0 / (h*h)));
    CSRSetValue(Dxx, N, N, (-1.0 / (h*h)));
    CSRSetValue(Dxx, N, N-1, (1.0 / (h*h)));

    CSRmatrix *Dxxxx = CSR_matrix_square(Dxx);
    CSRSetValue(Dxxxx, 0, 0, 1.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, 0, 1, -2.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, 0, 2, 1.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, 1, 0, -2.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, 1, 1, 5.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, 1, 2, -4.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, 1, 3, 1.0 / (h*h*h*h));

    CSRSetValue(Dxxxx, N, N, 1.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, N, N-1, -2.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, N, N-2, 1.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, N-1, N, -2.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, N-1, N-1, 5.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, N-1, N-2, -4.0 / (h*h*h*h));
    CSRSetValue(Dxxxx, N-1, N-3, 1.0 / (h*h*h*h));

    double fac = 1.0 / (1.0 + (sig0*k));

    CSRmatrix *tmp1 = CSR_create_eye(N+1);
    CSR_scalar_mult(tmp1, 2.0);
    CSRmatrix *tmp2 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp2, c*c*k*k);
    CSRmatrix *tmp3 = CSR_duplicate(Dxxxx);
    CSR_scalar_mult(tmp3, -kappa*kappa*k*k);
    CSRmatrix *tmp4 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp4, 2.0*k*sig1);

    CSRmatrix *tmp5 = CSR_matrix_add(tmp1, tmp2);
    CSRmatrix *tmp6 = CSR_matrix_add(tmp3, tmp4);
    B = CSR_matrix_add(tmp5, tmp6);
    CSR_scalar_mult(B, fac);

    CSR_free(tmp1);
    CSR_free(tmp2);
    CSR_free(tmp3);
    CSR_free(tmp4);
    CSR_free(tmp5);
    CSR_free(tmp6);

    tmp1 = CSR_create_eye(N+1);
    CSR_scalar_mult(tmp1, -(1.0 - (sig0 * k)));
    CSR_scalar_mult(Dxx, -2.0*k*sig1);
    C = CSR_matrix_add(tmp1, Dxx);
    CSR_scalar_mult(C, fac);

    CSR_free(tmp1);

    CSR_free(Dxxxx);
    CSR_free(Dxx);

    delete[] diag1;

    /* connection stuff */
    js0 = ((k*k) / (rho * (1.0 + (sig0*k)) * h));
    isc0 = CSRGetValue(C, 0, 0);
    isc1 = CSRGetValue(C, 1, 0);
    isb0 = CSRGetValue(B, 0, 0);
    isb1 = CSRGetValue(B, 1, 0);
    isb2 = CSRGetValue(B, 2, 0);

    /* for inputs */
    alpha = js0;

    logMessage(1, "Connection co-efficients: %f, %f, %f, %f, %f, %f", js0, isc0, isc1, isb0,
	       isb1, isb2);
}

void ComponentString::logMatrices()
{
    saveMatrix(B, "B");
    saveMatrix(C, "C");
}

ComponentString::~ComponentString()
{
    CSR_free(B);
    CSR_free(C);
}

void ComponentString::runTimestep(int n)
{
    CSR_matrix_vector_mult(B, u1, u);
    CSR_matrix_vector(C, u2, u, FALSE, OP_ADD); 

    runInputs(n, u, u1, u2);
}

// subtract them from the 2 doubles at l
void ComponentString::getConnectionValues(double *l)
{
    // IsB update
    l[0] -= ((u1[0] * isb0) + (u1[1] * isb1) + (u1[2] * isb2));
    l[1] -= ((u1[ss-1] * isb0) + (u1[ss-2] * isb1) + (u1[ss-3] * isb2));

    // IsC update
    l[0] -= ((u2[0] * isc0) + (u2[1] * isc1));
    l[1] -= ((u2[ss-1] * isc0) + (u2[ss-2] * isc1));
}

void ComponentString::setConnectionValues(double *l)
{
    u[0] += l[0] * js0;
    u[ss-1] += l[1] * js0;
}
