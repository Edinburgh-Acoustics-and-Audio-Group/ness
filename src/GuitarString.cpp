/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

/*
 * This class simulates a guitar string, with optional frets, backboard and
 * fingers. It's also used as the main component in the string and bar network
 * code (without the frets, backboard and fingers, but with ConnectionNet1 objects
 * connecting it to other strings and bars).
 *
 * The main state update is simple and usually pretty fast, even though it's just
 * using the basic CSR matrix multiplication routines. Dealing with collisions
 * between the string and the other objects is the difficult part. This uses a
 * Newton solver as in the Matlab version.
 *
 * The two main optimisations are:
 *
 *  - the matrix multplication M = Ic*Jc, which is very slow when done naively,
 *    is replaced by a much faster but much more complicated alternative. The
 *    matrices are analysed at startup (see mapMStructure) to determine which of
 *    the important elements of Ic and Jc actually influence each element of M.
 *    Then, in the main loop, this information is used to first subtract the
 *    influence of the old finger positions from M, and then to add the
 *    contribution of the new finger positions instead. Because most of the matrix
 *    doesn't change each timestep, this is much faster than recomputing the whole
 *    thing.
 *
 *  - instead of solving a large sparse linear system, rank reduction is used to
 *    reduce this down to a much smaller dense system. Typically only a few of the
 *    variables in the larger system are actually interdependent, so the resulting
 *    dense system is very small and can be solved very quickly by factorisation.
 *    This technique is used in two places within the collision update. There is
 *    also a sparse Jacobi solver as a fallback for when the reduced system is too
 *    large, but this is very rarely used in practise.
 *
 * There is no parallelisation in here as the domain is too small for GPU to give
 * much benefit, and the collision code makes vectorisation too awkward. However,
 * the separate strings typically run in parallel with each other on multiple CPU
 * cores.
 */

#include "GuitarString.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"
#include "Input.h"
#include "MathUtil.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
using namespace std;

#define EPSILON 2.2204460492503131e-16

#define MAX_DIRECT_SOLVE_SIZE 50

GuitarString::GuitarString(string name, double L, double E, double T, double r, double rho, double T60_0,
			   double T60_1000)
    : Component1D(name)
{
    int i, j;

    this->L = L;
    this->E = E;
    this->T = T;
    this->radius = r;
    this->rho = rho;
    this->T60_0 = T60_0;
    this->T60_1000 = T60_1000;

    initialised = false;

    prof = new Profiler(30);

    // calculate scalars
    GlobalSettings *gs = GlobalSettings::getInstance();
    double sr = gs->getSampleRate();
    k = 1.0 / sr;
    double A = M_PI * r * r;
    double I = 0.25 * M_PI * r * r * r * r;
    double c = sqrt(T / (rho*A));
    double kappa = sqrt(E * I / (rho * A));
    sig0 = 6.0 * log(10.0) / T60_0;
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

    alpha = (k*k) / (rho*A*h*(1.0+sig0*k));
    logMessage(1, "alpha=%.20f", alpha);

    allocateState(N-1);

    // compute update matrices
    double *diag1 = new double[N-1];
    for (i = 0; i < (N-1); i++) {
	diag1[i] = 0.0;
    }
    diag1[0] = -2.0 / (h*h);
    diag1[1] = 1.0 / (h*h);

    CSRmatrix *Dxx = CSR_sym_toeplitz(diag1, N-1);
    CSRmatrix *Dxxxx = CSR_matrix_square(Dxx);
    double fac = 1.0 / (1.0 + (sig0*k));
    
    CSRmatrix *tmp1 = CSR_create_eye(N-1);
    CSR_scalar_mult(tmp1, 2.0);
    CSRmatrix *tmp2 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp2, c*c*k*k);
    CSRmatrix *tmp3 = CSR_duplicate(Dxxxx);
    CSR_scalar_mult(tmp3, -kappa*kappa*k*k);
    CSRmatrix *tmp4 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp4, 2.0*sig1*k);

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
    CSR_scalar_mult(tmp1, -(1.0 - (sig0*k)));
    tmp2 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp2, -2.0*sig1*k);
    C = CSR_matrix_add(tmp1, tmp2);
    CSR_scalar_mult(C, fac);

    CSR_free(tmp1);
    CSR_free(tmp2);

    delete[] diag1;

    collisionsOn = false;

    bbss = 0;
    fretnum = 0;

    // sensible default finger parameters
    Mf = 0.005;
    Kf = 1e7;
    alphaf = 3.3;
    betaf = 100.0;

    energy = NULL;
    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps()];

	uu1 = new double[ss * 2];
	etmp = new double[ss * 2];

	betaf = 0.0;

	// generate P
	CSRmatrix *P11 = CSR_create_eye(N-1);
	CSR_scalar_mult(P11, 0.5 * (rho * A * h / (k*k)));

	tmp1 = CSR_create_eye(N-1);
	CSR_scalar_mult(tmp1, -0.5 * (rho * A * h / (k*k)));
	CSR_scalar_mult(Dxx, -0.25 * (T * h));
	CSR_scalar_mult(Dxxxx, 0.25 * (E * 0.25 * M_PI * r * r * r * r * h));
	tmp2 = CSR_matrix_add(tmp1, Dxx);
	CSRmatrix *P12 = CSR_matrix_add(tmp2, Dxxxx);

	CSR_free(tmp1);
	CSR_free(tmp2);

	P = (CSRmatrix*)malloc(sizeof(CSRmatrix));
	CSR_setup(P, 2 * ss, 2 * ss, (2 * ss) + (2 * (P12->rowStart[ss])));

	int pidx = 0;
	for (i = 0; i < ss; i++) {
	    // copy the single element of P11
	    P->values[pidx] = P11->values[P11->rowStart[i]];
	    P->colIndex[pidx] = i;
	    pidx++;

	    // copy all elements of P12
	    for (j = P12->rowStart[i]; j < P12->rowStart[i+1]; j++) {
		P->values[pidx] = P12->values[j];
		P->colIndex[pidx] = P12->colIndex[j] + ss;
		pidx++;
	    }
	    
	    P->rowStart[i+1] = pidx;
	}
	for (; i < (ss * 2); i++) {
	    // copy all elements of P12
	    for (j = P12->rowStart[i - ss]; j < P12->rowStart[i - ss + 1]; j++) {
		P->values[pidx] = P12->values[j];
		P->colIndex[pidx] = P12->colIndex[j];
		pidx++;
	    }
	    // copy the single element of P11
	    P->values[pidx] = P11->values[P11->rowStart[i - ss]];
	    P->colIndex[pidx] = i;
	    pidx++;

	    P->rowStart[i+1] = pidx;
	}

	CSR_free(P11);
	CSR_free(P12);
    }

    CSR_free(Dxxxx);
    CSR_free(Dxx);
}

void GuitarString::setBackboard(double b0, double b1, double b2)
{
    collisionsOn = true;
    logMessage(1, "b0=%f, b1=%f, b2=%f", b0, b1, b2);
    this->b0 = b0;
    this->b1 = b1;
    this->b2 = b2;
    bbss = ss;
}

void GuitarString::setFrets(vector<double> &fretpos, vector<double> &fretheight)
{
    int i;
    collisionsOn = true;
    fretnum = fretpos.size();
    logMessage(1, "Adding %d frets", fretnum);
    this->fretpos = new double[fretnum];
    this->fretheight = new double[fretnum];
    for (i = 0; i < fretnum; i++) {
	this->fretpos[i] = fretpos[i];
	this->fretheight[i] = fretheight[i];
    }
}

void GuitarString::setBarrierParams(double K, double alpha, double beta, int itnum, double tol)
{
    if (energy) {
	itnum = 50;
	tol = 0.0;
    }

    logMessage(1, "K=%f, alpha=%f, beta=%f, itnum=%d, tol=%f", K, alpha, beta, itnum, tol);
    this->Kb = K;
    this->alphab = alpha;
    this->betab = beta;
    this->itnum = itnum;
    this->tol = tol;
}

void GuitarString::setFingerParams(double Mf, double Kf, double alphaf, double betaf)
{
    if (energy) betaf = 0.0;

    logMessage(1, "Setting finger params: Mf=%f, Kf=%f, alphaf=%f, betaf=%f", Mf, Kf,
	       alphaf, betaf);
    this->Mf = Mf;
    this->Kf = Kf;
    this->alphaf = alphaf;
    this->betaf = betaf;
}

void GuitarString::addFinger(double uf0, double vf0, vector<double> *times, vector<double> *position,
			     vector<double> *force)
{
    logMessage(1, "Adding finger to string %s, uf0=%f, vf0=%f", name.c_str(), uf0, vf0);
    collisionsOn = true;
    Finger finger;
    finger.uf0 = uf0;
    finger.vf0 = vf0;
    finger.xf_int_prev = 0;

    finger.force = new BreakpointFunction(times->data(), force->data(), force->size(), k);
    finger.position = new BreakpointFunction(times->data(), position->data(), position->size(), k);

    fingers.push_back(finger);

    // breakpoint functions won't work if iterations get skipped
    Input::setFirstInputTimestep(0);
}


void GuitarString::setupCollisions()
{
    int i, j, k;

    if (collisionsOn) {
	int fn = fingers.size();
	Nc = bbss + fretnum + fn;

	logMessage(1, "bbss=%d, fretnum=%d, fn=%d, Nc=%d", bbss, fretnum, fn, Nc);

	Ic = (CSRmatrix *)malloc(sizeof(CSRmatrix));
	CSR_setup(Ic, Nc, ss + fn, bbss + (4 * fretnum) + fn + (ss*fn));
	Jc = (CSRmatrix *)malloc(sizeof(CSRmatrix));
	CSR_setup(Jc, ss + fn, Nc, bbss + (4 * fretnum) + fn + (ss*fn));

	betac = new double[Nc];
	Kc = new double[Nc];
	alphac = new double[Nc];
	coeffc = new double[Nc];

	eta1 = new double[Nc];
	eta2 = new double[Nc];
	g = new double[Nc];
	b = new double[Nc];
	r = new double[Nc];
	phi_ra = new double[Nc];
	R = new double[Nc];
	qc = new double[Nc];

	tmpvec = new double[Nc];
	Dinv = new double[Nc];

	maxa = new double[Nc];
	phi_a = new double[Nc];
	fac2 = new double[Nc];
	fac3 = new double[Nc];
	phi_prime = new double[Nc];
	maxrepsinv = new double[Nc];
	absreps = new double[Nc];
	notabsreps = new double[Nc];
	phidiff = new double[Nc];
	F = new double[Nc];
	temp = new double[Nc];
	D = new double[Nc];
	Dind = new int[Nc];
	Drev = new int[Nc];
	Fred = new double[Nc];
	temp2 = new double[Nc];
	temp3 = new double[Nc];

	utmp = new double[ss + fn];

	Mdense = new double[MAX_DIRECT_SOLVE_SIZE * MAX_DIRECT_SOLVE_SIZE];
	Lcrout = new double[MAX_DIRECT_SOLVE_SIZE * MAX_DIRECT_SOLVE_SIZE];
	Ucrout = new double[MAX_DIRECT_SOLVE_SIZE * MAX_DIRECT_SOLVE_SIZE];

	if (energy) {
	    eta = new double[Nc];
	}
	else {
	    eta = NULL;
	}

	// generate backboard portion of collision data
	for (i = 0; i < bbss; i++) {
	    CSRSetValue(Ic, i, i, 1.0);
	    CSRSetValue(Jc, i, i, alpha);

	    double xax = ((double)(i+1)) / ((double)(bbss+1));
	    b[i] = b0 + b1*xax + b2*xax*xax;
	}

	// handle frets
	for (i = 0; i < fretnum; i++) {
	    int x_int = (int)floor(fretpos[i] * ((double)(ss+1)));
	    double x_alpha = (fretpos[i] * ((double)(ss+1))) - ((double)x_int);

	    if ((x_int >= 1) && (x_int < ss)) {
		CSRSetValue(Ic, bbss + i, x_int - 1, 1.0 - x_alpha);
		CSRSetValue(Ic, bbss + i, x_int, x_alpha);

		CSRSetValue(Jc, x_int - 1, bbss + i, (1.0 - x_alpha) * alpha);
		CSRSetValue(Jc, x_int, bbss + i, x_alpha * alpha);
	    }
	    if (x_int < 1) {
		CSRSetValue(Ic, bbss + i, 0, x_alpha);
		CSRSetValue(Jc, 0, bbss + i, x_alpha * alpha);
	    }
	    if (x_int == ss) {
		CSRSetValue(Ic, bbss + i, x_int - 1, 1.0 - x_alpha);
		CSRSetValue(Jc, x_int - 1, bbss + i, (1.0 - x_alpha) * alpha);
	    }

	    b[bbss + i] = fretheight[i];
	}

	// handle fingers
	for (i = 0; i < fn; i++) {
	    // b
	    b[bbss + fretnum + i] = 0.0;

	    // Ic (extra identity)
	    CSRSetValue(Ic, bbss + fretnum + i, ss + i, 1.0);

	    // Ic (If section)
	    // set this to 1 and not 0 so that it survives the CSR_matrix_multiply and makes it
	    // into the structure of M
	    for (j = 0; j < ss; j++) {
		CSRSetValue(Ic, bbss + fretnum + i, j, 1.0);
	    }

	    // Jc (extra diagonal)
	    CSRSetValue(Jc, ss + i, bbss + fretnum + i, (this->k*this->k) / Mf);

	    // Jc (Jf section)
	    for (j = 0; j < ss; j++) {
		CSRSetValue(Jc, j, bbss + fretnum + i, 1.0);
	    }

	    // now store pointers into Ic and Jc for each value of xf_int, for use later on
	    fingers[i].IcLocs = new double*[ss];
	    fingers[i].JcLocs = new double*[ss];

	    fingers[i].IcM = new vector<int>[ss];
	    fingers[i].JcM = new vector<int>[ss];
	    
	    // Ic locations are in row bbss + fretnum + i, col 0 - (ss-1)
	    for (j = Ic->rowStart[bbss + fretnum + i]; j < Ic->rowStart[bbss + fretnum + i + 1]; j++) {
		if (Ic->colIndex[j] < ss) {
		    fingers[i].IcLocs[Ic->colIndex[j]] = &Ic->values[j];
		}
	    }

	    // Jc locations are in row 0 - (ss-1), col bbss + fretnum + i
	    for (j = 0; j < ss; j++) {
		for (k = Jc->rowStart[j]; k < Jc->rowStart[j+1]; k++) {
		    if (Jc->colIndex[k] == (bbss + fretnum + i)) {
			fingers[i].JcLocs[j] = &Jc->values[k];
		    }
		}
	    }
	}

	M = CSR_matrix_multiply(Ic, Jc);

	IMQ = CSR_duplicate(M);
	W = CSR_duplicate(M);

	// initialise vectors
	for (i = 0; i < Nc; i++) {
	    betac[i] = betab;
	    if (energy) betac[i] = 0.0;
	    Kc[i] = Kb;
	    alphac[i] = alphab;
	    coeffc[i] = Kc[i] / (1.0 + alphac[i]);

	    eta1[i] = 0.0;
	    eta2[i] = 0.0;
	    g[i] = 0.0;
	    r[i] = 0.0;
	    phi_ra[i] = 0.0;
	    R[i] = 0.0;
	    qc[i] = 0.0;
	}
	for (i = 0; i < fingers.size(); i++) {
	    betac[bbss + fretnum + i] = betaf;
	    Kc[bbss + fretnum + i] = Kf;
	    alphac[bbss + fretnum + i] = alphaf;
	    coeffc[bbss + fretnum + i] = Kf / (1.0 + alphaf);

	    // also clear If entries for this finger
	    for (j = 0; j < ss; j++) {
		fingers[i].IcLocs[j][0] = 0.0;
		fingers[i].JcLocs[j][0] = 0.0;
	    }
	}

	// finger setup
	if (fingers.size() > 0) {
	    // extend state arrays and matrices for finger state
	    // we don't change ss, so functions for getting input/output positions should still work the same
	    double *newu, *newu1, *newu2;
	    newu = new double[ss + fingers.size()];
	    newu1 = new double[ss + fingers.size()];
	    newu2 = new double[ss + fingers.size()];
	    memcpy(newu, u, ss * sizeof(double));
	    memcpy(newu1, u1, ss * sizeof(double));
	    memcpy(newu2, u2, ss * sizeof(double));
	    delete[] u;
	    delete[] u1;
	    delete[] u2;
	    u = newu;
	    u1 = newu1;
	    u2 = newu2;

	    CSRmatrix *newB, *newC;
	    newB = (CSRmatrix *)malloc(sizeof(CSRmatrix));
	    newC = (CSRmatrix *)malloc(sizeof(CSRmatrix));
	    CSR_setup(newB, ss + fingers.size(), ss + fingers.size(), B->rowStart[ss] + fingers.size());
	    CSR_setup(newC, ss + fingers.size(), ss + fingers.size(), C->rowStart[ss] + fingers.size());
	    for (i = 0; i < ss; i++) {
		for (j = B->rowStart[i]; j < B->rowStart[i+1]; j++) {
		    newB->colIndex[j] = B->colIndex[j];
		    newB->values[j] = B->values[j];
		}

		for (j = C->rowStart[i]; j < C->rowStart[i+1]; j++) {
		    newC->colIndex[j] = C->colIndex[j];
		    newC->values[j] = C->values[j];
		}

		newB->rowStart[i+1] = B->rowStart[i+1];
		newC->rowStart[i+1] = C->rowStart[i+1];
	    }
	    j = newB->rowStart[ss];
	    k = newC->rowStart[ss];
	    for (; i < (ss + fingers.size()); i++) {
		newB->values[j] = 2.0;
		newB->colIndex[j] = i;
		j++;
		newB->rowStart[i+1] = j;

		newC->values[k] = -1.0;
		newC->colIndex[k] = i;
		k++;
		newC->rowStart[i+1] = k;
	    }
	    CSR_free(B);
	    B = newB;
	    CSR_free(C);
	    C = newC;

	    // initialise finger states
	    for (i = 0; i < fingers.size(); i++) {
		u[ss + i] = 0.0;
		u1[ss + i] = fingers[i].uf0 + (k * fingers[i].vf0);
		u2[ss + i] = fingers[i].uf0;
	    }

	    // work out how M is influenced by finger elements of Ic and Jc
	    mapMStructure();

	    // propagate the 0s from If into M
	    CSR_matrix_multiply_reuse(Ic, Jc, M);
	}
    }
}

/*
 * Scan through M, working out which elements are influenced by the "finger"
 * elements of Ic and Jc, so that we can update just those elements and save a
 * lot of time.
 */
void GuitarString::mapMStructure()
{
    int i, j, k, l;
    real sum;
    int threshold = bbss + fretnum;

    // loop over M as if we were doing a matrix multiply
    /* loop over rows of result matrix (also rows of Ic) */
    for (i = 0; i < M->nrow; i++) {
	/* loop over columns of result matrix (also columns of Jc) */
	for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
	    int col = M->colIndex[j];

	    if ((i >= threshold) || (col >= threshold)) {
		/* need to do dot product of row i of Ic and column col of Jc */
		/* loop over non-zeroes in row i of Ic */
		for (k = Ic->rowStart[i]; k < Ic->rowStart[i+1]; k++) {
		    /* column of this value gives us a row to search in Jc */
		    int row = Ic->colIndex[k];
		    for (l = Jc->rowStart[row]; l < Jc->rowStart[row+1]; l++) {
			/* is there a value at this column? */
			if (Jc->colIndex[l] == col) {
			    if (row < ss) {
				if (i >= threshold) {
				    fingers[i - threshold].IcM[row].push_back(j);
				    fingers[i - threshold].IcM[row].push_back(l);
				    //printf("Element %d of If affects element %d of M (multiplied by element %d of Jc)\n", row, j, l);
				}
				// "else" here so that we don't have duplicate effects going on
				else if (col >= threshold) {
				    fingers[col - threshold].JcM[row].push_back(j);
				    fingers[col - threshold].JcM[row].push_back(k);
				    //printf("Element %d of Jf affects element %d of M (multiplied by element %d of Ic)\n", row, j, k);
				}
			    }
			    break;
			}
		    }
		}
	    }
	}
    }
}


GuitarString::~GuitarString()
{
    int i;
    CSR_free(B);
    CSR_free(C);
    
    logMessage(1, "String %s profile: %s\n", name.c_str(), prof->print().c_str());
    delete prof;

    if ((collisionsOn) && (initialised)) {
	CSR_free(Ic);
	CSR_free(Jc);
	CSR_free(M);
	CSR_free(W);

	delete[] betac;
	delete[] Kc;
	delete[] alphac;
	delete[] coeffc;
	delete[] eta1;
	delete[] eta2;
	delete[] g;
	delete[] b;
	delete[] r;
	delete[] phi_ra;
	delete[] R;
	delete[] qc;
	delete[] tmpvec;
	delete[] Dinv;
	delete[] utmp;

	if (fretnum) {
	    delete[] fretpos;
	    delete[] fretheight;
	}

	CSR_free(IMQ);
	delete[] maxa;
	delete[] phi_a;
	delete[] fac2;
	delete[] fac3;
	delete[] phi_prime;
	delete[] maxrepsinv;
	delete[] absreps;
	delete[] notabsreps;
	delete[] phidiff;
	delete[] F;
	delete[] temp;
	delete[] D;
	delete[] Dind;
	delete[] Drev;
	delete[] Fred;
	delete[] temp2;
	delete[] temp3;

	delete[] Mdense;
	delete[] Lcrout;
	delete[] Ucrout;

	for (i = 0; i < fingers.size(); i++) {
	    delete fingers[i].force;
	    delete fingers[i].position;
	    delete[] fingers[i].IcLocs;
	    delete[] fingers[i].JcLocs;
	    delete[] fingers[i].IcM;
	    delete[] fingers[i].JcM;
	}

	if (eta) delete[] eta;
    }

    if (energy) {
	delete[] energy;
	CSR_free(P);
	delete[] uu1;
	delete[] etmp;
    }
}

static void du(double *vec, int sz)
{
    int i;
    for (i = 0; i < sz; i++) {
	printf("%d: %.20f\n", i, vec[i]);
    }
}

void GuitarString::runTimestep(int n)
{
    int i, j;

    if (!initialised) {
	setupCollisions();
	initialised = true;
    }

    prof->start(0);
    CSR_matrix_vector_mult(B, u1, u);
    CSR_matrix_vector(C, u2, u, FALSE, OP_ADD);
    prof->end(0);

    prof->start(1);
    runInputs(n, u, u1, u2);

    // finger excitations
    if (!energy) {
	for (i = 0; i < fingers.size(); i++) {
	    double ff = ((k*k) / Mf) * fingers[i].force->getValue();
	    u[ss + i] -= ff;
	    fingers[i].force->next();
	}
    }
    prof->end(1);

    if (collisionsOn) {
	double *tmp;

	prof->start(2);
	// finger collision setup
	for (i = 0; i < fingers.size(); i++) {
	    // remove old finger influence from M
	    int xf_int_prev = fingers[i].xf_int_prev;
	    if (xf_int_prev < 1) {
		// handle Ic influence
		for (j = 0; j < fingers[i].IcM[xf_int_prev].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int_prev][j]] -=
			Jc->values[fingers[i].IcM[xf_int_prev][j+1]] * fingers[i].IcLocs[xf_int_prev][0];
		}
		// handle Jc influence
		for (j = 0; j < fingers[i].JcM[xf_int_prev].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int_prev][j]] -=
			Ic->values[fingers[i].JcM[xf_int_prev][j+1]] * fingers[i].JcLocs[xf_int_prev][0];
		}
	    }
	    else if (xf_int_prev == ss) {
		// handle Ic influence
		for (j = 0; j < fingers[i].IcM[xf_int_prev-1].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int_prev-1][j]] -=
			Jc->values[fingers[i].IcM[xf_int_prev-1][j+1]] * fingers[i].IcLocs[xf_int_prev-1][0];
		}
		// handle Jc influence
		for (j = 0; j < fingers[i].JcM[xf_int_prev-1].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int_prev-1][j]] -=
			Ic->values[fingers[i].JcM[xf_int_prev-1][j+1]] * fingers[i].JcLocs[xf_int_prev-1][0];
		}
	    }
	    else {
		// handle Ic influence
		for (j = 0; j < fingers[i].IcM[xf_int_prev-1].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int_prev-1][j]] -=
			Jc->values[fingers[i].IcM[xf_int_prev-1][j+1]] * fingers[i].IcLocs[xf_int_prev-1][0];
		}
		for (j = 0; j < fingers[i].IcM[xf_int_prev].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int_prev][j]] -=
			Jc->values[fingers[i].IcM[xf_int_prev][j+1]] * fingers[i].IcLocs[xf_int_prev][0];
		}
		// handle Jc influence
		for (j = 0; j < fingers[i].JcM[xf_int_prev-1].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int_prev-1][j]] -=
			Ic->values[fingers[i].JcM[xf_int_prev-1][j+1]] * fingers[i].JcLocs[xf_int_prev-1][0];
		}
		for (j = 0; j < fingers[i].JcM[xf_int_prev].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int_prev][j]] -=
			Ic->values[fingers[i].JcM[xf_int_prev][j+1]] * fingers[i].JcLocs[xf_int_prev][0];
		}
	    }
	}
	prof->end(2);

	prof->start(3);
	for (i = 0; i < fingers.size(); i++) {
	    // now compute new Ic and Jc
	    double xf = fingers[i].position->getValue();
	    fingers[i].position->next();

	    if (energy) xf = fingers[i].uf0;

	    int xf_int = (int)floor(xf * (double)(ss+1));
	    double xf_alpha = (xf * (double)(ss+1)) - (double)xf_int;
	    int xf_int_prev = fingers[i].xf_int_prev;
	    
	    // clear the part used last time
	    if (xf_int_prev < 1) {
		fingers[i].IcLocs[xf_int_prev][0] = 0.0;
		fingers[i].JcLocs[xf_int_prev][0] = 0.0;
	    }
	    else if (fingers[i].xf_int_prev == ss) {
		fingers[i].IcLocs[xf_int_prev-1][0] = 0.0;
		fingers[i].JcLocs[xf_int_prev-1][0] = 0.0;
	    }
	    else {
		fingers[i].IcLocs[xf_int_prev-1][0] = 0.0;
		fingers[i].JcLocs[xf_int_prev-1][0] = 0.0;
		fingers[i].IcLocs[xf_int_prev][0] = 0.0;
		fingers[i].JcLocs[xf_int_prev][0] = 0.0;
	    }

	    if ((xf_int >= 1) && (xf_int < ss)) {
		fingers[i].IcLocs[xf_int-1][0] = -(1.0 - xf_alpha);
		fingers[i].IcLocs[xf_int][0] = -(xf_alpha);
		
		fingers[i].JcLocs[xf_int-1][0] = -(1.0 - xf_alpha) * alpha;
		fingers[i].JcLocs[xf_int][0] = -(xf_alpha) * alpha;
	    }
	    if (xf_int < 1) {
		fingers[i].IcLocs[0][0] = -xf_alpha;
		fingers[i].JcLocs[0][0] = -xf_alpha * alpha;
	    }
	    if (xf_int == ss) {
		fingers[i].IcLocs[xf_int-1][0] = -(1.0 - xf_alpha);
		fingers[i].JcLocs[xf_int-1][0] = -(1.0 - xf_alpha) * alpha;
	    }

	    fingers[i].xf_int_prev = xf_int;
	}
	prof->end(3);

	prof->start(23);
	for (i = 0; i < fingers.size(); i++) {
	    int xf_int = fingers[i].xf_int_prev;

	    // add new finger influence to M
	    if (xf_int < 1) {
		// handle Ic influence
		for (j = 0; j < fingers[i].IcM[xf_int].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int][j]] +=
			Jc->values[fingers[i].IcM[xf_int][j+1]] * fingers[i].IcLocs[xf_int][0];
		}
		// handle Jc influence
		for (j = 0; j < fingers[i].JcM[xf_int].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int][j]] +=
			Ic->values[fingers[i].JcM[xf_int][j+1]] * fingers[i].JcLocs[xf_int][0];
		}
	    }
	    else if (xf_int == ss) {
		// handle Ic influence
		for (j = 0; j < fingers[i].IcM[xf_int-1].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int-1][j]] +=
			Jc->values[fingers[i].IcM[xf_int-1][j+1]] * fingers[i].IcLocs[xf_int-1][0];
		}
		// handle Jc influence
		for (j = 0; j < fingers[i].JcM[xf_int-1].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int-1][j]] +=
			Ic->values[fingers[i].JcM[xf_int-1][j+1]] * fingers[i].JcLocs[xf_int-1][0];
		}
	    }
	    else {
		// handle Ic influence
		for (j = 0; j < fingers[i].IcM[xf_int-1].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int-1][j]] +=
			Jc->values[fingers[i].IcM[xf_int-1][j+1]] * fingers[i].IcLocs[xf_int-1][0];
		}
		for (j = 0; j < fingers[i].IcM[xf_int].size(); j += 2) {
		    M->values[fingers[i].IcM[xf_int][j]] +=
			Jc->values[fingers[i].IcM[xf_int][j+1]] * fingers[i].IcLocs[xf_int][0];
		}
		// handle Jc influence
		for (j = 0; j < fingers[i].JcM[xf_int-1].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int-1][j]] +=
			Ic->values[fingers[i].JcM[xf_int-1][j+1]] * fingers[i].JcLocs[xf_int-1][0];
		}
		for (j = 0; j < fingers[i].JcM[xf_int].size(); j += 2) {
		    M->values[fingers[i].JcM[xf_int][j]] +=
			Ic->values[fingers[i].JcM[xf_int][j+1]] * fingers[i].JcLocs[xf_int][0];
		}
	    }
	}
	prof->end(23);
	
	prof->start(4);
	for (i = 0; i < (ss + fingers.size()); i++) {
	    utmp[i] = u[i] - u2[i];
	}
	// g = Ic*(u-u2)
	CSR_matrix_vector_mult(Ic, utmp, g);

	// swap eta2 and eta1
	tmp = eta2;
	eta2 = eta1;
	eta1 = tmp;

	// eta1 = b - Ic*u1
	for (i = 0; i < Nc; i++) {
	    eta1[i] = b[i];
	}
	CSR_matrix_vector(Ic, u1, eta1, FALSE, OP_SUB);

	// get max(eta1)
	double maxeta1 = -1000000000.0;
	for (i = 0; i < Nc; i++) {
	    if (eta1[i] > maxeta1) maxeta1 = eta1[i];
	}
	prof->end(4);

	prof->start(5);
	int qnum = 0;
	if (maxeta1 > 0.0) {
	    // prepare qc
	    for (i = 0; i < Nc; i++) {
		Drev[i] = -1;
		if (eta1[i] > 0.0) {
		    qc[i] = ((1.0 / (2.0 * k)) * betac[i] * Kc[i] * pow(eta1[i], alphac[i]));
		    if ((qc[i] > EPSILON) || (qc[i] < -EPSILON)) {
			Drev[i] = qnum;
			Dind[qnum++] = i; // store mapping arrays for rank reduction later on
		    }
		}
		else {
		    qc[i] = 0.0;
		}
	    }

	    // do r = -(Idc0 + M*Qc) \ g
	    // where Idc0 is an identity matrix of size Nc, and Qc is a diagonal matrix version of qc
	    if (qnum <= MAX_DIRECT_SOLVE_SIZE) {

		// handle case of no non-zeroes in qc
		if (qnum == 0) {		    
		    for (i = 0; i < Nc; i++) {
			r[i] = -g[i];
		    }
		}
		else {
		    // small system, use a direct solve
		    
		    // map g down to small vector (in temp)
		    for (i = 0; i < qnum; i++) {
			temp[i] = g[Dind[i]];
		    }
		    
		    // compute Mdense
		    // first clear the section we need
		    for (i = 0; i < (qnum*qnum); i++) Mdense[i] = 0.0;
		    
		    // now fill in the values
		    for (i = 0; i < qnum; i++) {
			for (j = M->rowStart[Dind[i]]; j < M->rowStart[Dind[i]+1]; j++) {
			    if (Drev[M->colIndex[j]] >= 0) {
				// found one
				Mdense[(i*qnum) + Drev[M->colIndex[j]]] = M->values[j] * qc[M->colIndex[j]];
				if (i == Drev[M->colIndex[j]]) {
				    Mdense[(i*qnum) + Drev[M->colIndex[j]]] += 1.0;
				}
			    }
			}
		    }
		    
		    if (qnum == 1) {
			// handle (common) case of single unknown
			temp2[0] = temp[0] / Mdense[0];
		    }
		    else {
			// obtain LU decomposition
			if (!croutDecomposition(Mdense, Lcrout, Ucrout, qnum)) {
			    logMessage(5, "Error obtaining Crout decomposition (size %d, iter %de)", qnum, n);
			}
			
			// solve (temp3 is scratch buffer, temp2 is result)
			croutSolve(Lcrout, Ucrout, temp, temp2, temp3, qnum);
		    }
		    
		    // map back to full size vector (r) and negate in the process
		    for (i = 0; i < Nc; i++) {
			if (Drev[i] >= 0) {
			    r[i] = temp2[Drev[i]];
			}
			else {
			    r[i] = g[i];
			    for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
				if (Drev[M->colIndex[j]] >= 0) {
				    if (i != M->colIndex[j]) {
					r[i] -= M->values[j] * qc[M->colIndex[j]] * temp2[Drev[M->colIndex[j]]];
				    }
				}
			    }
			}
			r[i] = -r[i];
		    }
		}
	    }
	    else {
		if (!jacobiSolve1(r, g, M, qc, 1e-9, 500)) {
		    logMessage(1, "External Jacobi solver failed to converge (iteration %d)", n);
		}
	    }
	}
	else {
	    for (i = 0; i < Nc; i++) {
		qc[i] = 0.0;
		r[i] = -g[i];
	    }
	    qnum = 0;
	}
	prof->end(5);

	prof->start(6);
	double amax = -1000000000.0, ramax = -1000000000.0;
	for (i = 0; i < Nc; i++) {
	    if (eta2[i] > amax) amax = eta2[i];
	    if ((r[i] + eta2[i]) > ramax) ramax = r[i] + eta2[i];
	}
	prof->end(6);

	if ((amax > 0.0) || (ramax > 0.0)) {
	    newtonSolve(r, eta2, g, M, qc, Kc, alphac, coeffc, betac, Nc, itnum, tol, phi_ra, R, n, qnum);
	}
	else {
	    prof->start(8);
	    for (i = 0; i < Nc; i++) {
		R[i] = qc[i] * r[i];
	    }
	    prof->end(8);
	}
	
	prof->start(7);
	// update main state vector with result of collisions
	CSR_matrix_vector(Jc, R, u, FALSE, OP_ADD);
	prof->end(7);
    }
}

void GuitarString::newtonSolve(double *r, double *eta2, double *g, CSRmatrix *M, double *qc, double *Kc,
			       double *alphac, double *coeffc, double *betac, int Nc, int itnum, double tol,
			       double *phi_ra, double *R, int nnn, int qnum)
{
    int i, j, nn;
    int maxannz;
    int maxrannz;
    int phidiffnnz;
    int Rnum;
    double resid;

    prof->start(9);
    // compute maxa, maxannz and qnum
    maxannz = 0;
    for (i = 0; i < Nc; i++) {
	if (eta2[i] > 0.0) {
	    maxa[i] = eta2[i];
	    maxannz++;
	}
	else maxa[i] = 0.0;
    }

    // compute phi_a, fac2 and fac3 vectors
    if (maxannz > 0) {
	for (i = 0; i < Nc; i++) {
	    double maxpow = 0.0;
	    if (maxa[i] > 0.0) {
		maxpow = pow(maxa[i], alphac[i] - 1.0);
	    }
	    phi_a[i] = coeffc[i] * (maxpow * maxa[i] * maxa[i]);
	    fac2[i] = (0.5 * alphac[i] * (alphac[i] + 1.0) * Kc[i] * maxpow);
	    fac3[i] = Kc[i] * maxpow * maxa[i];
	}
    }
    else {
	for (i = 0; i < Nc; i++) {
	    phi_a[i] = 0.0;
	    fac2[i] = 0.0;
	    fac3[i] = 0.0;
	}
    }
    prof->end(9);

    prof->start(10);
    // compute IMQ
    // only works if IMQ and M have identical structure!
    if (qnum > 0) {
	// IMQ = I + M*q
	for (i = 0; i < Nc; i++) {
	    for (j = IMQ->rowStart[i]; j < IMQ->rowStart[i+1]; j++) {
		if (IMQ->colIndex[j] == i) {
		    IMQ->values[j] = 1.0 + (M->values[j] * qc[i]);
		}
		else {
		    IMQ->values[j] = M->values[j] * qc[IMQ->colIndex[j]];
		}
	    }
	}
    }
    else {
	// IMQ = I
	for (i = 0; i < Nc; i++) {
	    for (j = IMQ->rowStart[i]; j < IMQ->rowStart[i+1]; j++) {
		if (IMQ->colIndex[j] == i) {
		    IMQ->values[j] = 1.0;
		}
		else {
		    IMQ->values[j] = 0.0;
		}
	    }
	}
    }
    prof->end(10);

    // main iteration loop
    for (nn = 1; nn <= itnum; nn++) {
	prof->start(11);
	// compute maxra - reusing maxa vector for this
	maxrannz = 0;
	for (i = 0; i < Nc; i++) {
	    if ((r[i] + eta2[i]) > 0.0) {
		maxa[i] = r[i] + eta2[i];
		maxrannz++;
	    }
	    else {
		maxa[i] = 0.0;
	    }
	}

	// compute phi_ra and phi_prime
	if (maxrannz > 0) {
	    for (i = 0; i < Nc; i++) {
		double maxraalpha = 0;
		if (maxa[i] > 0.0) {
		    maxraalpha = pow(maxa[i], alphac[i]);
		}
		phi_ra[i] = coeffc[i] * maxa[i] * maxraalpha;
		phi_prime[i] = Kc[i] * maxraalpha;
	    }
	}
	else {
	    for (i = 0; i < Nc; i++) {
		phi_ra[i] = 0.0;
		phi_prime[i] = 0.0;
	    }
	}

	// compute maxrepsinv, absreps, notabsreps, phidiff
	phidiffnnz = 0;
	for (i = 0; i < Nc; i++) {
	    if (fabs(r[i]) > EPSILON) {
		maxrepsinv[i] = 1.0 / fabs(r[i]);
		absreps[i] = 1.0;
		notabsreps[i] = 0.0;
	    }
	    else {
		maxrepsinv[i] = 1.0 / EPSILON;
		absreps[i] = 0.0;
		notabsreps[i] = 1.0;
	    }
	    phidiff[i] = phi_ra[i] - phi_a[i];
	    if ((phidiff[i] > EPSILON) || (phidiff[i] < -EPSILON)) phidiffnnz++;
	}

	// compute R and Rnum
	Rnum = 0;
	if ((phidiffnnz > 0) || (maxannz > 0)) {
	    for (i = 0; i < Nc; i++) {
		double sgnr = r[i] > 0.0 ? 1.0 : -1.0;
		R[i] = absreps[i] * sgnr * phidiff[i] * maxrepsinv[i] + notabsreps[i] * fac3[i];
		if ((R[i] > EPSILON) || (R[i] < -EPSILON)) Rnum++;
	    }
	}
	else {
	    for (i = 0; i < Nc; i++) {
		R[i] = 0.0;
	    }
	}
	prof->end(11);

	if (Rnum == 0) {
	    prof->start(12);
	    // do r = -IMQ\b
	    if (!jacobiSolve2(r, g, IMQ, 1e-9, 500)) {
		logMessage(3, "Short-circuit Jacobi solver failed to converge!");
	    }
	    for (i = 0; i < Nc; i++) {
		r[i] = -r[i];
	    }
	    prof->end(12);

	    // FIXME: does this path ever get taken when IMQ is initialised to identity above?
	    // if so, avoid having to call the solver at all
	    break;
	}

	// surely this is always true? if Rnum is not > 0, the "break" above happens
	if ((qnum > 0) || (Rnum > 0)) {
	    prof->start(13);
	    // compute F = IMQ*r + M*R + b
	    // only works if IMQ and M have identical structure!
	    for (i = 0; i < Nc; i++) {
		F[i] = g[i];
		for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
		    F[i] += M->values[j] * R[M->colIndex[j]];
		    F[i] += IMQ->values[j] * r[M->colIndex[j]];
		}
	    }
	    prof->end(13);

	    // compute temp and D, and get indices of non-zeroes in D
	    prof->start(14);
	    int Dnum = 0;
	    for (i = 0; i < Nc; i++) {
		temp[i] = absreps[i] * (r[i]*phi_prime[i] - phidiff[i]) * maxrepsinv[i] * maxrepsinv[i] +
		    notabsreps[i] * fac2[i];
		D[i] = temp[i] + qc[i];
		Drev[i] = -1;
		if ((D[i] > EPSILON) || (D[i] < -EPSILON)) {
		    Drev[i] = Dnum; // store reverse mapping from original indices to only non-zeroes
		    Dind[Dnum++] = i;
		}
	    }
	    prof->end(14);

	    prof->start(15);
	    // compute Fred = Dred * F;
	    for (i = 0; i < Dnum; i++) {
		Fred[i] = D[Dind[i]] * F[Dind[i]];
	    }
	    prof->end(15);
		
	    if (Dnum <= MAX_DIRECT_SOLVE_SIZE) {
		prof->start(16);
		// small system, use a direct solve
		// compute Mdense
		// first clear the section we need
		for (i = 0; i < (Dnum*Dnum); i++) Mdense[i] = 0.0;
		
		// now fill in the values
		for (i = 0; i < Dnum; i++) {
		    for (j = M->rowStart[Dind[i]]; j < M->rowStart[Dind[i]+1]; j++) {
			if (Drev[M->colIndex[j]] >= 0) {
			    // found one
			    Mdense[(i*Dnum) + Drev[M->colIndex[j]]] = M->values[j] * D[Dind[i]];
			    if (i == Drev[M->colIndex[j]]) {
				Mdense[(i*Dnum) + Drev[M->colIndex[j]]] += 1.0;
			    }
			}
		    }
		}
		prof->end(16);

		prof->start(17);
		if (Dnum == 1) {
		    // handle (common) case of single unknown
		    temp2[0] = Fred[0] / Mdense[0];
		}
		else {
		    // obtain LU decomposition
		    if (!croutDecomposition(Mdense, Lcrout, Ucrout, Dnum)) {
			logMessage(5, "Error obtaining Crout decomposition (size %d, iter %d)", Dnum, nnn);
		    }
		    
		    // solve (temp3 is scratch buffer, temp2 is result)
		    croutSolve(Lcrout, Ucrout, Fred, temp2, temp3, Dnum);
		}
		prof->end(17);
	    }
	    else {
		// large system, iterative (Jacobi) solve
		
		prof->start(18);
		// compute W matrix
		int wpos = 0;
		W->nrow = Dnum;
		W->ncol = Dnum;
		// loop over rows of W (also rows of Dred)
		for (i = 0; i < Dnum; i++) {
		    // each row of Dred contains one value, at column Dind[i]
		    // check that row of M for any values that haven't been cropped out of Mred
		    for (j = M->rowStart[Dind[i]]; j < M->rowStart[Dind[i]+1]; j++) {
			if (Drev[M->colIndex[j]] >= 0) {
			    // found one
			    W->values[wpos] = M->values[j] * D[Dind[i]];
			    // add one if diagonal
			    if (i == Drev[M->colIndex[j]]) {
				W->values[wpos] += 1.0;
			    }
			    W->colIndex[wpos] = Drev[M->colIndex[j]];
			    wpos++;
			}
		    }
		    
		    // remember fill in W->rowStart!
		    W->rowStart[i+1] = wpos;
		}
		prof->end(18);
		
		// linear system solve: temp2 = W \ Fred;
		prof->start(19);
		if (!jacobiSolve2(temp2, Fred, W, 1e-9, 500)) {
		    logMessage(5, "Main Jacobi solver failed to converge (size %d) at iteration %d,%d!", Dnum, nnn, nn);
		}
		prof->end(19);
	    }
		
	    // temp3 = Mred*temp2
	    prof->start(20);
	    for (i = 0; i < Nc; i++) {
		temp3[i] = 0.0;
		for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
		    if (Drev[M->colIndex[j]] >= 0) {
			temp3[i] += M->values[j] * temp2[Drev[M->colIndex[j]]];
		    }
		}
	    }
	    prof->end(20);

	    // update r and compute relative residual
	    prof->start(21);
	    resid = 0.0;
	    for (i = 0; i < Nc; i++) {
		double newr = r[i] - F[i] + temp3[i];
		resid += (newr - r[i]) * (newr - r[i]);
		r[i] = newr;
	    }
	    prof->end(21);
	}

	// break out if tolerance reached
	if (resid < (tol*tol)) break;
    }

    if ((nn > itnum) && (!energy)) {
	// don't print the warning when energy is on because then the tolerance is set
	// to 0.0 so we don't expect convergence!
	logMessage(3, "Newton solver failed to converge for iteration %d!", nnn);
    }

    prof->start(22);
    // R = R + Q*r
    for (i = 0; i < Nc; i++) {
	R[i] = R[i] + qc[i] * r[i];
    }
    prof->end(22);
}

void GuitarString::swapBuffers(int n)
{
    int i;

    // do energy check
    if (energy) {
	// basic string energy
	for (i = 0; i < ss; i++) {
	    uu1[i] = u[i];
	    uu1[i+ss] = u1[i];
	}
	CSR_matrix_vector_mult(P, uu1, etmp);
	energy[n] = 0.0;
	for (i = 0; i < (ss*2); i++) {
	    energy[n] += uu1[i] * etmp[i];
	}

	// add finger state part of energy
	for (i = 0; i < fingers.size(); i++) {
	    double fe1 = ((0.5 * Mf * u[ss+i]) / (k*k)) - ((0.5 * Mf * u1[ss+i]) / (k*k));
	    double fe2 = ((0.5 * Mf * u1[ss+i]) / (k*k)) - ((0.5 * Mf * u[ss+i]) / (k*k));
	    energy[n] += (fe1 * u[ss+i]) + (fe2 * u1[ss+i]);
	}

	// add collision part of energy
	if (collisionsOn) {
	    for (i = 0; i < Nc; i++) {
		eta[i] = b[i];
	    }
	    CSR_matrix_vector(Ic, u, eta, FALSE, OP_SUB);
	    for (i = 0; i < Nc; i++) {
		double maxeta = 0.0;
		if (eta[i] > 0.0) maxeta = eta[i];
		double phi = (Kc[i] / (alphac[i] + 1.0)) * pow(maxeta, alphac[i] + 1.0);
		maxeta = 0.0;
		if (eta1[i] > 0.0) maxeta = eta1[i];
		double phi1 = (Kc[i] / (alphac[i] + 1.0)) * pow(maxeta, alphac[i] + 1.0);
		energy[n] += 0.5 * (phi + phi1);
	    }
	}
    }

    Component::swapBuffers(n);
}



// solves r = IMQ \ b
bool GuitarString::jacobiSolve2(double *r, double *b, CSRmatrix *IMQ, double jtol, int jmaxit)
{
    int i, j, k;
    double resid = 1000.0;
    double newr;

    // compute Dinv
    for (i = 0; i < IMQ->nrow; i++) {
	for (j = IMQ->rowStart[i]; j < IMQ->rowStart[i+1]; j++) {
	    if (IMQ->colIndex[j] == i) {
		Dinv[i] = 1.0 / IMQ->values[j];
		break;
	    }
	}
	// start with solution of r = b
	r[i] = b[i];
    }

    // main Jacobi method loop
    jtol = jtol * jtol;
    k = 0;
    while ((resid > jtol) && (k < jmaxit)) {
	// tmpvec = R.r
	// where R is off-diagonals of IMQ
	for (i = 0; i < IMQ->nrow; i++) {
	    tmpvec[i] = 0.0;
	    for (j = IMQ->rowStart[i]; j < IMQ->rowStart[i+1]; j++) {
		if (IMQ->colIndex[j] != i) {
		    tmpvec[i] += IMQ->values[j] * r[IMQ->colIndex[j]];
		}
	    }
	}

	resid = 0.0;

	// r = Dinv * (b - tmpvec)
	for (i = 0; i < IMQ->nrow; i++) {
	    newr = Dinv[i] * (b[i] - tmpvec[i]);
	    // compute relative residual
	    resid += (newr - r[i]) * (newr - r[i]);
	    r[i] = newr;
	}
	k++;
    }
    return (k < jmaxit);
}

// Solves r = -(Idc0 + M*Qc) \ g using the Jacobi method
// Idc0 is an identity matrix so no need to pass this
// Qc is a diagonal matrix version of the vector qc
bool GuitarString::jacobiSolve1(double *r, double *g, CSRmatrix *M, double *qc, double jtol, int jmaxit)
{
    int i, j, k;
    double resid = 1000.0;
    double newr;

    // compute Dinv
    for (i = 0; i < Nc; i++) {
	for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
	    if (M->colIndex[j] == i) {
		Dinv[i] = 1.0 / -(1.0 + M->values[j] * qc[i]);
		break;
	    }
	}

	// start with solution of -g
	r[i] = -g[i];
    }

    // main Jacobi method loop
    k = 0;
    while ((resid > jtol) && (k < jmaxit)) {
	// tmpvec = R.r
	// where R is off-diagonals of -M*Qc (Idc0 doesn't have any off-diagonals)
	for (i = 0; i < Nc; i++) {
	    tmpvec[i] = 0.0;
	    for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
		if (M->colIndex[j] != i) {
		    tmpvec[i] -= M->values[j] * r[M->colIndex[j]] * qc[M->colIndex[j]];
		}
	    }
	}

	resid = 0.0;

	// r = Dinv * (g - tmpvec)
	for (i = 0; i < Nc; i++) {
	    newr = Dinv[i] * (g[i] - tmpvec[i]);
	    // compute relative residual
	    resid += (newr - r[i]) * (newr - r[i]);
	    r[i] = newr;
	}
	k++;
    }
    return (k < jmaxit);
}

void GuitarString::logMatrices()
{
    saveMatrix(B, "B");
    saveMatrix(C, "C");
    if (collisionsOn) {
	saveMatrix(Ic, "Ic");
	saveMatrix(Jc, "Jc");
	saveMatrix(M, "M");
    }
}

double *GuitarString::getEnergy()
{
    return energy;
}
