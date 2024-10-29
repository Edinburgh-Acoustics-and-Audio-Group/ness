/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "Fretboard.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "MathUtil.h"
#include "Input.h"

#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;

Fretboard::Fretboard(string name, double L, double E, double T, double r, double rho, double T60_0,
		     double T60_1000, int fretnum, double b0, double b1, double fretheight, double Kb,
		     double alphab, double betab, int itnum)
    : Component1D(name)
{
    int i;

    this->L = L;
    this->E = E;
    this->T = T;
    radius = r;
    this->rho = rho;
    this->T60_0 = T60_0;
    this->T60_1000 = T60_1000;

    /* compute scalars */
    double SR = GlobalSettings::getInstance()->getSampleRate();
    double k = 1.0 / SR;
    A = M_PI * r*r;
    double I = 0.25 * M_PI * r*r*r*r;
    double c = sqrt(T / (rho * A));
    double kappa = sqrt((E * I) / (rho * A));
    double sig_0 = (6.0 * log(10.0)) / T60_0;
    double z1 = (-c*c + sqrt(c*c*c*c + 4.0*kappa*kappa * (2.0*M_PI*1000.0)*(2.0*M_PI*1000.0))) /
	(2.0 * kappa * kappa);
    if (kappa == 0.0) {
	z1 = ((2.0*M_PI*1000.0) * (2.0*M_PI*1000.0)) / (c*c);
    }
    if (c == 0.0) {
	z1 = (2.0 * M_PI * 1000.0) / kappa;
    }
    double sig_1 = ((6.0 * log(10.0)) / z1) * ((1.0 / T60_1000) - (1.0 / T60_0));

    h = sqrt(0.5 * (c*c*k*k + 4.0*sig_1*k +
			   sqrt((c*c*k*k + 4.0*sig_1*k)*(c*c*k*k + 4.0*sig_1*k) + 16.0*kappa*kappa*k*k)));
    int N = (int)floor(L / h);
    h = L / (double)N;

    logMessage(1, "k=%f, A=%f, I=%f, c=%f, kappa=%f, sig0=%f, z1=%f, sig1=%f, N=%d, h=%f",
	       k, A, I, c, kappa, sig_0, z1, sig_1, N, h);

    /* allocate state arrays */
    allocateState(N - 1);

    /* create update matrices */
    double *diag1 = new double[N - 1];
    for (i = 0; i < (N - 1); i++) {
	diag1[i] = 0.0;
    }
    diag1[0] = -2.0 / (h*h);
    diag1[1] = 1.0 / (h*h);

    CSRmatrix *Dxx = CSR_sym_toeplitz(diag1, N-1);
    CSRmatrix *Dxxxx = CSR_matrix_square(Dxx);
    fac = 1.0 / (1.0 + (sig_0*k));

    CSRmatrix *tmp1 = CSR_create_eye(N-1);
    CSR_scalar_mult(tmp1, 2.0);
    CSRmatrix *tmp2 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp2, c*c*k*k);
    CSRmatrix *tmp3 = CSR_duplicate(Dxxxx);
    CSR_scalar_mult(tmp3, -kappa*kappa*k*k);
    CSRmatrix *tmp4 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp4, 2.0*k*sig_1);

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

    tmp1 = CSR_create_eye(N-1);
    CSR_scalar_mult(tmp1, -(1.0 - (sig_0 * k)));
    CSR_scalar_mult(Dxx, -2.0*k*sig_1);
    C = CSR_matrix_add(tmp1, Dxx);
    CSR_scalar_mult(C, fac);

    CSR_free(tmp1);

    CSR_free(Dxxxx);
    CSR_free(Dxx);

    delete[] diag1;

    // set alpha for inputs
    this->alpha = (k*k*fac) / (rho*A*h);

    /* initialise frets */
    // compute Nb
    Nb = (N - 1) + fretnum;

    // allocate b
    b = new double[Nb + 1];

    // create Ib (and I0). Fill in b
    I0 = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(I0, Nb + 1, N, (N-1) + (2*fretnum) + N);

    // add identity matrix at top left, for fretboard
    for (i = 0; i < (N-1); i++) {
	CSRSetValue(I0, i, i, 1.0);
	double xax = (((double)i+1)*h);
	b[i] = b0 + b1 * xax * xax;
    }
    // now handle frets themselves
    for (i = 0; i < fretnum; i++) {
	double fretpos = L * (1.0 - pow(2.0, -((((double)i)+1.0) / 12.0)));
	int xfret_ind = (int)floor(fretpos / h);
	double xfret_frac = (fretpos / h) - ((double)xfret_ind);

	CSRSetValue(I0, (N-1)+i, xfret_ind-1, (1.0 - xfret_frac));
	CSRSetValue(I0, (N-1)+i, xfret_ind, xfret_frac);

	b[(N-1) + i] = fretheight;
    }
    // add entire dense row for the finger. Don't use 0s here as it causes entries to not
    // be created where they're needed in M0
    for (i = 0; i < (N-1); i++) {
	CSRSetValue(I0, Nb, i, 1.0);
    }
    CSRSetValue(I0, Nb, N-1, 1.0);

    // create Jb (and J0)
    J0 = CSR_transpose(I0);
    CSR_scalar_mult(J0, fac * ((k*k) / (rho*A*h)));
    CSRSetValue(J0, (N-1), Nb, 1.0); // will fill in actual value dependent on Mf later

    // create M0 (non-zero structure)
    tmp1 = CSR_matrix_multiply(I0, J0);
    M0 = removeDiagonalEntries(tmp1);
    CSR_free(tmp1);


    /* initialise finger */
    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    ff = new double[NF];
    xf_int = new int[NF];
    xf_frac = new double[NF];

    memset(ff, 0, NF * sizeof(double));
    memset(xf_int, 0, NF * sizeof(int));
    memset(xf_frac, 0, NF * sizeof(double));

    // extend state arrays
    delete[] u;
    delete[] u1;
    delete[] u2;
    u = new double[ss + 1];
    u1 = new double[ss + 1];
    u2 = new double[ss + 1];
    memset(u, 0, (ss+1) * sizeof(double));
    memset(u1, 0, (ss+1) * sizeof(double));
    memset(u2, 0, (ss+1) * sizeof(double));

    // extend state update matrices
    B = extendMatrix(B, 2.0);
    C = extendMatrix(C, -1.0);

    // extend b and create K, alphan, beta arrays
    b[Nb] = 0.0;
    K = new double[Nb+1];
    alphan = new double[Nb+1];
    beta = new double[Nb+1];
    for (i = 0; i < Nb; i++) {
	K[i] = Kb;
	alphan[i] = alphab;
	beta[i] = betab;
    }

    // allocate temporaries for main loop
    this->r = new double[Nb+1];
    eta1 = new double[Nb+1];
    eta2 = new double[Nb+1];
    g = new double[Nb+1];
    q = new double[Nb+1];

    phi_ra = new double[Nb+1];
    Mdiag = new double[Nb+1];
    R = new double[Nb+1];

    // create Newton solver
    newton = newtonSolverCreate(itnum, 1e-9, 500, 1e-9, M0);

    // with this component, things can happen before the first standard "Input",
    // so make sure earlier timesteps aren't skipped
    Input::setFirstInputTimestep(0);
}

Fretboard::~Fretboard()
{
    CSR_free(B);
    CSR_free(C);
    CSR_free(I0);
    CSR_free(J0);
    CSR_free(M0);

    delete[] b;
    delete[] K;
    delete[] alphan;
    delete[] beta;
    delete[] r;
    delete[] eta1;
    delete[] eta2;
    delete[] g;
    delete[] q;

    delete[] ff;
    delete[] xf_int;
    delete[] xf_frac;

    delete[] phi_ra;
    delete[] Mdiag;
    delete[] R;

    newtonSolverFree(newton);
}

// FIXME: eventually try to move finger stuff into a new class in Input hierarchy and make it
// available to other components when relevant
void Fretboard::setFingerParams(double Mf, double Kf, double alphaf, double betaf, double uf0, double vf0,
				vector<double> *fingertime, vector<double> *fingerpos,
				vector<double> *fingerforce)
{
    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    double SR = GlobalSettings::getInstance()->getSampleRate();
    double k = 1.0 / SR;

    int N = ss + 1;
    int Nb = J0->ncol - 1;

    int i;

    CSRSetValue(J0, (N-1), Nb, (k*k)/Mf);

    K[Nb] = Kf;
    alphan[Nb] = alphaf;
    beta[Nb] = betaf;

    // set initial finger position
    u1[ss] = uf0 + k*vf0;
    u2[ss] = uf0;

    // compute ff
    double *xq = new double[NF];
    int fingerexclen = fingertime->size();

    for (i = 0; i < NF; i++) {
	xq[i] = ((double)i) * k;
    }

    interp1(fingertime->data(), fingerforce->data(), xq, fingerexclen, NF, ff);
    for (i = 0; i < NF; i++) {
	// scale ff
	ff[i] *= -((k*k) / Mf);
    }

    // compute xfpos
    double *xfpos = new double[NF];

    interp1(fingertime->data(), fingerpos->data(), xq, fingerexclen, NF, xfpos);
    for (i = 0; i < NF; i++) {
	// compute xf_int and xf_frac
	xf_int[i] = (int)floor((xfpos[i] * L) / h);
	xf_frac[i] = ((xfpos[i] * L) / h) - ((double)xf_int[i]);
    }
    delete[] xfpos;
    delete[] xq;
}


/*
 * Expands a CSRmatrix by one element in each dimension and adds a new
 * element in the bottom right corner
 */
CSRmatrix *Fretboard::extendMatrix(CSRmatrix *old, double val)
{
    // create new matrix
    CSRmatrix *mat = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    int nnz = old->rowStart[old->nrow];
    CSR_setup(mat, old->nrow + 1, old->ncol + 1, nnz + 1);

    // copy existing arrays
    memcpy(mat->rowStart, old->rowStart, (old->nrow + 1) * sizeof(int));
    memcpy(mat->colIndex, old->colIndex, nnz * sizeof(int));
    memcpy(mat->values, old->values, nnz * sizeof(double));

    // insert new bottom right value
    mat->rowStart[mat->nrow] = nnz + 1;
    mat->colIndex[nnz] = mat->ncol - 1;
    mat->values[nnz] = val;

    // free old and return new
    CSR_free(old);
    return mat;
}

void Fretboard::runTimestep(int n)
{
    int i, j;

    /* do basic string update */
    CSR_matrix_vector_mult(B, u1, u);
    CSR_matrix_vector(C, u2, u, FALSE, OP_ADD);

    runInputs(n, u, u1, u2);

    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    double SR = GlobalSettings::getInstance()->getSampleRate();
    double k = 1.0 / SR;

    // do finger excitation update
    u[ss] += ff[n];

    // do barrier update
    // update J0 - dense final column (column Nb)
    // should be all 0s except around xf_int
    int Nb = J0->ncol - 1;
    double p = xf_frac[n];
    double f2 = -fac * ((k*k) / (rho*A*h));
    for (i = 0; i < (J0->nrow-1); i++) {
	for (j = J0->rowStart[i]; j < J0->rowStart[i+1]; j++) {
	    if (J0->colIndex[j] == Nb) {
		if (i == (xf_int[n] - 2)) {
		    J0->values[j] = (((p-1.0)*p*(p-2.0)) / (-6.0)) * f2;
		}
		else if (i == (xf_int[n] - 1)) {
		    J0->values[j] = (((p-1.0)*(p+1.0)*(p-2.0)) / 2.0) * f2;
		}
		else if (i == (xf_int[n])) {
		    J0->values[j] = ((p*(p-2.0)*(p+1.0)) / (-2.0)) * f2;
		}
		else if (i == (xf_int[n] + 1)) {
		    J0->values[j] = ((p*(p-1.0)*(p+1.0)) / 6.0) * f2;
		}
		else {
		    J0->values[j] = 0.0;
		}
	    }
	}
    }

    // update I0
    double plast = xf_frac[0];
    int ilast = xf_int[0];
    if (n > 0) { plast = xf_frac[n-1]; ilast = xf_int[n-1]; }
    double pnext = xf_frac[NF-1];
    int inext = xf_int[NF-1];
    if (n < (NF-1)) { pnext = xf_frac[n+1]; inext = xf_int[n+1]; }

    for (i = I0->rowStart[Nb]; i < (I0->rowStart[Nb+1] - 1); i++) {
	I0->values[i] = 0.0;
	int ci = I0->colIndex[i];
	if (ci == (inext-1)) {
	    I0->values[i] += -0.5 * (1.0 - pnext);
	}
	if (ci == (inext)) {
	    I0->values[i] += -0.5 * pnext;
	}
	if (ci == (ilast-1)) {
	    I0->values[i] += -0.5 * (1.0 - plast);
	}
	if (ci == (ilast)) {
	    I0->values[i] += -0.5 * plast;
	}
    }

    // M0 = I0 * J0
    matrixMultiplyReuse(I0, J0, M0, Mdiag);

    // g = I0*(u-u2)
    // eta1 = b - I0*u1
    // eta2 = b - I0*u2
    // q = (1/(2*k))*beta.*K.*(max(eta1,0)).^alpha
    // r = -g
    for (i = 0; i < I0->nrow; i++) {
	g[i] = 0.0;
	eta1[i] = b[i];
	eta2[i] = b[i];
	for (j = I0->rowStart[i]; j < I0->rowStart[i+1]; j++) {
	    g[i] += I0->values[j] * (u[I0->colIndex[j]] - u2[I0->colIndex[j]]);
	    eta1[i] -= I0->values[j] * u1[I0->colIndex[j]];
	    eta2[i] -= I0->values[j] * u2[I0->colIndex[j]];
	}

	if (eta1[i] > 0.0) {
	    q[i] = (1.0 / (2.0*k)) * beta[i] * K[i] * pow(eta1[i], alphan[i]);
	}
	else {
	    q[i] = 0.0;
	}
	r[i] = -g[i];
    }

    // do Newton solve
    newtonSolverVector(newton, r, eta2, g, M0, Mdiag, q, K, alphan, phi_ra, R);

    // update u from result (u = u + J0*R)
    CSR_matrix_vector(J0, R, u, 0, 1);
}

/*
 * Returns a matrix that is the same as the input matrix but with the diagonal entries removed
 */
CSRmatrix *Fretboard::removeDiagonalEntries(CSRmatrix *mat)
{
    int i, j, idx;
    CSRmatrix *result;

    // count the non-diagonals
    int nnz = 0;
    for (i = 0; i < mat->nrow; i++) {
	for (j = mat->rowStart[i]; j < mat->rowStart[i+1]; j++) {
	    if (mat->colIndex[j] != i) nnz++;
	}
    }

    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(result, mat->nrow, mat->ncol, nnz);

    idx = 0;
    result->rowStart[0] = 0;
    for (i = 0; i < mat->nrow; i++) {
	for (j = mat->rowStart[i]; j < mat->rowStart[i+1]; j++) {
	    if (mat->colIndex[j] != i) {
		result->values[idx] = mat->values[j];
		result->colIndex[idx] = mat->colIndex[j];
		idx++;
	    }
	}
	result->rowStart[i+1] = idx;
    }
    return result;
}

/*
 * Multiplies in1 and in2 to give out. The non-zero structure must already be present in the
 * output matrix or the behaviour will be undefined. The diagonal entries are stored into the
 * vector diag instead of into the output matrix.
 */
void Fretboard::matrixMultiplyReuse(CSRmatrix *in1, CSRmatrix *in2, CSRmatrix *out, double *diag)
{
    int i, j, k, l;
    double sum;

    /* loop over rows of result matrix (also rows of matrix 1) */
    for (i = 0; i < out->nrow; i++) {
	/* do diagonal entry first */
	sum = 0.0;
	/* loop over non-zeroes in row i of matrix 1 */
	for (k = in1->rowStart[i]; k < in1->rowStart[i+1]; k++) {
	    /* column of this value gives us a row to search in matrix 2 */
	    int row = in1->colIndex[k];
	    for (l = in2->rowStart[row]; l < in2->rowStart[row+1]; l++) {
		/* is there a value at the right column? */
		if (in2->colIndex[l] == i) {
		    sum += in2->values[l] * in1->values[k];
		    break;
		}
	    }
	}
	diag[i] = sum;

	/* loop over other columns of result matrix (also columns of matrix 2) */
	for (j = out->rowStart[i]; j < out->rowStart[i+1]; j++) {
	    int col = out->colIndex[j];
	    /* need to do dot product of row i of matrix 1 and column col of matrix 2 */
	    sum = 0.0;
	    /* loop over non-zeroes in row i of matrix 1 */
	    for (k = in1->rowStart[i]; k < in1->rowStart[i+1]; k++) {
		/* column of this value gives us a row to search in matrix 2 */
		int row = in1->colIndex[k];
		for (l = in2->rowStart[row]; l < in2->rowStart[row+1]; l++) {
		    /* is there a value at this column? */
		    if (in2->colIndex[l] == col) {
			sum += in2->values[l] * in1->values[k];
			break;
		    }
		}
	    }
	    out->values[j] = sum;
	}	
    }
}

void Fretboard::logMatrices()
{
    saveMatrix(B, "B");
    saveMatrix(C, "C");
    saveMatrix(I0, "I0");
    saveMatrix(J0, "J0");
    saveMatrix(M0, "M0");
}
