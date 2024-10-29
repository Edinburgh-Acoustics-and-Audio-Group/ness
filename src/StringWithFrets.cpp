/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "StringWithFrets.h"
#include "Logger.h"

#include <cmath>
using namespace std;

StringWithFrets::StringWithFrets(string name, double L, double rho, double T, double E,
				 double r, double T60_0, double T60_1000, double xc1,
				 double yc1, double xc2, double yc2, int numfrets,
				 double fretheight, double backboardheight,
				 double backboardvar)
    : ComponentString(name, L, rho, T, E, r, T60_0, T60_1000, xc1, yc1, xc2, yc2)
{
    int i;
    int N = ss - 1;

    logMessage(1, "Barrier parameters: %d, %f, %f, %f", numfrets, fretheight, backboardheight,
	       backboardvar);

    // allocate Ic matrix
    int matw = (N - 1) + numfrets;
    logMessage(1, "Matrix width for barrier: %d", matw);

    Ic = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Ic, (N+1), matw, (N-1) + (2 * numfrets));

    // fill in the identity values
    for (i = 0; i < (N-1); i++) {
	CSRSetValue(Ic, i+1, i, 1.0);
    }

    // add in fret values
    for (i = 0; i < numfrets; i++) {
	double xfret = L * (1.0 - pow(2.0, (-((i+1)/12.0))));
	int xpos_int = (int)floor((((double)N)*xfret) / L);
	double xpos_alpha = ((((double)N) * xfret) / L) - ((double) xpos_int);
	
	CSRSetValue(Ic, xpos_int, N - 1 + i, 1.0 - xpos_alpha);
	CSRSetValue(Ic, xpos_int + 1, N - 1 + i, xpos_alpha);
    }
    
    // generate btot as well
    btot = new double[matw];
    for (i = 0; i < (N-1); i++) {
	double xback = (i + 1) * h;
	double bback = (backboardheight - (backboardvar * xback * xback));
	btot[i] = bback;
    }
    for (; i < matw; i++) {
	btot[i] = fretheight;
    }

    // allocate the other vectors required
    a = new double[matw];
    b = new double[matw];
    this->r = new double[matw];
    R = new double[matw];
    F = new double[matw];
    temp = new double[matw];

    utmp = new double[N+1];

    // generate Jc
    Jc = CSR_duplicate(Ic);
    CSR_scalar_mult(Jc, (k*k) / (rho * (1.0 + (sig0*k)) * h));

    // might as well transpose Ic here as it's only used in transposed form
    CSRmatrix *tmp = CSR_transpose(Ic);
    CSR_free(Ic);
    Ic = tmp;

    // generate M
    M = CSR_matrix_multiply(Ic, Jc);
    J = CSR_duplicate(M);

    // create PCG solver
    pcg = pcgCreate(matw, NULL, NULL, 1e-12, 500);
}

void StringWithFrets::logMatrices()
{
    ComponentString::logMatrices();
    saveMatrix(Ic, "Ic");
    saveMatrix(Jc, "Jc");
    saveMatrix(M, "M");
    saveMatrix(J, "J");
    saveVector(btot, Ic->nrow, "btot");
}

StringWithFrets::~StringWithFrets()
{
    CSR_free(Ic);
    CSR_free(Jc);
    CSR_free(M);
    CSR_free(J);

    delete[] btot;
    delete[] a;
    delete[] b;
    delete[] r;
    delete[] R;
    delete[] F;
    delete[] temp;
    delete[] utmp;

    pcgFree(pcg);
}

void StringWithFrets::runTimestep(int n)
{
    ComponentString::runTimestep(n);

    int i;
    int matw = Ic->nrow;

    // get b
    for (i = 0; i < ss; i++) {
	utmp[i] = u[i] - u2[i];
    }
    CSR_matrix_vector_mult(Ic, utmp, b);

    // get a: (btot - Ic * u2)
    memcpy(a, btot, matw*sizeof(double));
    CSR_matrix_vector(Ic, u2, a, FALSE, OP_SUB);
    
    // r = -b
    for (i = 0; i < matw; i++) {
	r[i] = -b[i];
    }

    // do Newton's Method!
    newtonsMethod();

    // update u based on result
    CSR_matrix_vector(Jc, R, u, FALSE, OP_ADD);    
}

void StringWithFrets::newtonsMethod()
{
    int i, j, k;
    int N = M->nrow;
    double coeff = K / (alphaNewton + 1.0);
    double eps = 2.220446049250313e-16;
    double phi_ra, phi_a, phi_prime;
    double ra;
    double rapow;
    double apow;

    // loop over iterations
    for (i = 0; i < iter; i++) {

	// compute R, temp, and start computing F
	for (j = 0; j < N; j++) {
	    // compute the phi_* values:
	    //   phi_ra = coeff * pow(max(r[j]+a[j], 0.0), (alpha+1.0));
	    //   phi_a = coeff * pow(max(a[j], 0.0), (alpha+1.0));
	    //   phi_prime = K * pow(max(r[j]+a[j], 0.0), alpha);
	    // try to save as many expensive pow calls as we can
	    if (fabs(r[j]) > eps) {
		ra = r[j] + a[j];
		if (ra > 0.0) {
		    rapow = pow(ra, alphaNewton);
		    phi_ra = coeff * rapow * ra;
		    phi_prime = K * rapow;
		}
		else {
		    phi_ra = 0.0;
		    phi_prime = 0.0;
		}
		if (a[j] > 0.0) {
		    phi_a = coeff * pow(a[j], (alphaNewton+1.0));
		}
		else {
		    phi_a = 0.0;
		}

		R[j] = (phi_ra - phi_a) / r[j];
		temp[j] = (r[j] * phi_prime - phi_ra + phi_a) / (r[j] * r[j]);
	    }
	    else {
		if (a[j] > 0.0) {
		    apow = pow(a[j], (alphaNewton - 1.0));
		    R[j] = K * apow * a[j];
		    temp[j] = 0.5 * alphaNewton * (alphaNewton + 1.0) * K * apow;
		}
		else {
		    R[j] = 0.0;
		    temp[j] = 0.0;
		}
	    }
	    F[j] = r[j] + b[j];
	}
	// finish computing F
	CSR_matrix_vector(M, R, F, FALSE, OP_ADD);

	// compute J
	// J is M with its columns scaled by temp, plus the identity matrix
	// safe to do this because J and M share exactly the same structure
	for (j = 0; j < J->nrow; j++) {
	    for (k = J->rowStart[j]; k < J->rowStart[j+1]; k++) {
		J->values[k] = M->values[k] * temp[J->colIndex[k]];
		if (J->colIndex[k] == j) {
		    J->values[k] += 1.0;
		}
	    }
	}

	// solve J\F into temp
	memset(temp, 0, N * sizeof(double));

	// FIXME: may be able to use a simpler method for this
	pcgSolve(pcg, J, temp, F);

	// update r
	for (j = 0; j < N; j++) {
	    r[j] = r[j] - temp[j];
	}
    }
}
