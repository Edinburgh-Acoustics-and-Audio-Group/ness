/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

/*
 * This class implements a non-linear modal plate. For smaller plates it is
 * usually faster than the finite difference version in PlateEmbedded, though it
 * doesn't support all the features of PlateEmbedded (only rectangular plates, no
 * embeddings, etc.).
 *
 * The main state update is actually really simple; it's a few lines of basic
 * matrix and vector operations in Matlab. This class is made complicated by two
 * factors:
 *
 *  1. The startup code that generates H1 is quite involved.
 *  2. It's highly optimised. CUDA, multi-threading and vectorisation (AVX) are all
 *     used to speed it up.
 *
 * Because the main part of the startup code (Hcalc_unstable) can take a long time,
 * the H1 matrix is cached on disk after it's generated. Then if a plate with the
 * same dimensions and number of modes is used again later (to run a different
 * score file on the same instrument, for example) the previously computed H1 can
 * be loaded from disk instead of having to calculate it again.
 *
 * H1 is sparse, but because it has relatively few zero values compared to most
 * sparse matrices, the Matlab version, the CUDA version and the unoptimised C++
 * version all treat it as if it were dense. However, the AVX version takes
 * advantage of the sparseness. Although the matrix structure is much more random
 * looking than the strictly banded matrices used in the finite difference codes,
 * there do tend to be relatively few patterns of non-zero values that come up
 * again and again throughout. By permuting the rows of the matrix so that all of
 * the rows with the same pattern are consecutive, and re-ordering the data in
 * memory, the matrix multiplication can be vectorised with 4 rows being processed
 * at a time (see ModalPlate_avx.cpp for the actual vector kernel). Of course, the
 * opposite permutation then has to be applied to get the result back into the
 * correct order.
 *
 * The optimiseForAVX method sets up all of this. It analyses H1 making a list of
 * all the non-zero patterns present, generates the forward and reverse
 * permutations, and condenses and re-orders the values in H1 ready for the vector
 * kernel.
 *
 * The multithreading support consists of wrapping up the update of each row type
 * (as implemented in runRowType) into a separate Task (getParallelTasks returns
 * a list of all the tasks). The higher levels of the code take care of farming
 * these tasks out to all the available CPU cores. There is a single serial Task
 * that runs after all the parallel ones complete; this calls finishUpdate, which
 * does the whole rest of the timestep update after the multiplication by H1.
 *
 * The GPU version is quite simple in comparison and treats the operation as a
 * basic dense-matrix-by-vector multiply. For larger plates with more modes this
 * is usually faster than the CPU version, but it's limited by the size of the GPU
 * memory, as H1 can get pretty big.
 */

#include "ModalPlate.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "MathUtil.h"
#include "SettingsManager.h"
#include "Input.h"

#include "TaskWholeComponent.h"
#include "TaskModalPlateRow.h"
#include "TaskModalPlateEnd.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
using namespace std;

#define DD 100

void getCacheDirectory(char *buf, int bufsize);

ModalPlate::ModalPlate(string name, int A, double Lx, double Ly, double h,
		       double nu, double Young, double rho, double loss1,
		       double loss2, bool halfNyquist)
    : Component(name)
{
    int i;
    GlobalSettings *gs = GlobalSettings::getInstance();

    logMessage(1, "Creating ModalPlate: modes=%d, size=%fx%f, h=%f, nu=%f, E=%f, rho=%f", A, Lx, Ly, h, nu, Young, rho);

    this->Lx = Lx;
    this->Ly = Ly;

    this->A = A;
    this->nu = nu;
    this->Young = Young;
    this->rho = rho;
    this->h = h;

    this->loss1 = loss1;
    this->loss2 = loss2;

    // first calculate DIM from sample rate
    double *ov = omega_calculator();
    double fs = GlobalSettings::getInstance()->getSampleRate();
    for (i = 0; i < (DD*DD); i++) {
	if (ov[i*3] < (2.0*fs)) {
	    DIM = (i+1);
	}
    }
    delete[] ov;

    if (halfNyquist) DIM /= 2;
    
    // Remember state size must be set equal to DIM!
    ss = DIM;

    // allocate state arrays
    u = new double[ss];
    u1 = new double[ss];
    u2 = new double[ss];

    memset(u, 0, ss * sizeof(double));
    memset(u1, 0, ss * sizeof(double));
    memset(u2, 0, ss * sizeof(double));

    // perform eigencalc
    double *coeff1 = new double[A*A*A*A*sizeof(double)];
    int S = eigencalcUnstable(A, Lx, Ly, coeff1);

    logMessage(1, "Performed eigencalcUnstable, S=%d", S);	

    H1 = new double[DIM*DIM*S];

    // see if cached H1 exists
    char cachedir[1000];
    getCacheDirectory(cachedir, 1000);
    char cachefile[1000];
    sprintf(cachefile, "%smodalplate_h1_%d_%d_%f_%f.bin", cachedir, DIM, A, Lx, Ly);
    ifstream fi(cachefile, ios::in | ios::binary);
    if (fi.good()) {
	logMessage(1, "Loading H1 from cache");

	// found cache file, load it
	fi.read((char *)H1, DIM*DIM*S*sizeof(double));
	fi.close();
    }
    else {
	Hcalc_unstable(coeff1, DIM, A, Lx, Ly, S, H1);	
	logMessage(1, "Performed Hcalc_unstable");

	// write to cache for next time
	ofstream fo(cachefile, ios::out | ios::binary);
	if (fo.good()) {
	    fo.write((const char *)H1, DIM*DIM*S*sizeof(double));
	    fo.close();
	}
    }
    delete[] coeff1;
	
    C = new double[DIM];
    C1 = new double[DIM];
    C2 = new double[DIM];
    Cnorm = new double[DIM];

    instr_def(fs);

    logMessage(1, "Performed instr_def, sample rate=%f", fs);

    this->A = S;

    // scale H1
    double nr = sqrt(Lx*Ly / 4.0);
    double h1scale = sqrt((nr*nr)/rho*Young/2.0);
    for (i = 0; i < (DIM*DIM*this->A); i++) {
	H1[i] *= h1scale;
    }

    // allocate temporary buffers
    t1 = new double[S*DIM];
    t2 = new double[S];
    G = new double[DIM];

    memset(G, 0, DIM * sizeof(double));

    logMessage(1, "Final mode counts: A=%d, DIM=%d", this->A, this->DIM);

    H1vals = NULL;

    if (gs->getAVXEnabled()) {
	optimiseForAVX();
    }

    Input::setFirstInputTimestep(2);

#ifdef USE_GPU
    gpuModalPlate = NULL;
#endif
}

ModalPlate::~ModalPlate()
{
    delete[] H1;
    delete[] C;
    delete[] C1;
    delete[] C2;
    delete[] Cnorm;
    delete[] ov;
    delete[] t1;
    delete[] t2;
    delete[] G;

    if (H1vals) {
	// delete AVX data
	delete[] H1vals;
	delete[] rowTypeIndex;
	delete[] rowInfo;
	delete[] rowTypeCounts;
	delete[] rowPerm;
	delete[] reversePerm;
	delete[] t1perm;
    }

#ifdef USE_GPU
    if (gpuModalPlate) {
	delete gpuModalPlate;
	// state arrays will be freed by gpuModalPlate, don't let Component destructor free them
	u = NULL;
	u1 = NULL;
	u2 = NULL;
    }
#endif
}

void ModalPlate::optimiseForAVX()
{
    int i, j, k, l, m;

    // array of all possible row types (with 1 for value, 0 for 0)
    char *rowTypes;

    int numRows;
    int rt;
    
    // analyse H1, look for different row types
    numRowTypes = 0;
    numRows = A * DIM;

    // maximum possible number of row types is one per row
    rowTypes = new char[numRows * DIM];
    rowTypeIndex = new int[numRows];

    // loop over all rows and find the type for each
    for (i = 0; i < numRows; i++) {
	rt = -1;
	/* loop over existing row types and look for a match */
	for (j = 0; j < numRowTypes; j++) {
	    rt = j;

	    /* loop over this row type and see if it matches the row */
	    for (k = 0; k < DIM; k++) {
		if (((rowTypes[(j*DIM)  + k]) && (H1[(k*numRows)+i] == 0.0)) ||
		    ((!rowTypes[(j*DIM) + k]) && (H1[(k*numRows)+i] != 0.0))) {
		    /* mismatch */
		    rt = -1;
		    break;
		}
	    }

	    if (rt >= 0) break; /* found it! */
	}

	if (rt >= 0) {
	    /* it matches an existing type */
	    rowTypeIndex[i] = rt;
	}
	else {
	    /* add a new type */
	    j = numRowTypes;
	    numRowTypes++;

	    for (k = 0; k < DIM; k++) {
		if (H1[(k*numRows)+i] == 0.0) {
		    rowTypes[(j*DIM)+k] = 0;
		}
		else {
		    rowTypes[(j*DIM)+k] = 1;
		}
	    }
	    rowTypeIndex[i] = j;
	}
    }

    logMessage(1, "Found %d row types, proceeding with AVX optimisation", numRowTypes);

    // generate forward and reverse permutations
    rowPerm = new int[numRows];
    reversePerm = new int[numRows];
    rowTypeCounts = new int[numRowTypes];

    k = 0;
    // loop over types
    for (i = 0; i < numRowTypes; i++) {
	rowTypeCounts[i] = 0;
	// put all the rows of this type together
	for (j = 0; j < numRows; j++) {
	    if (rowTypeIndex[j] == i) {
		/* found another one of current type */
		rowPerm[k] = j;
		reversePerm[j] = k;
		k++;
		rowTypeCounts[i]++;
	    }
	}
    }

    // allocate storage for permuted version of t1
    t1perm = new double[A * DIM];

    // re-order and condense H1 for AVX
    H1vals = new double[numRows * DIM];
    k = 0;
    for (i = 0; i < numRowTypes; i++) {
	/* loop over the rows in 4s */
	j = 0;
	for (j = 0; j < (rowTypeCounts[i] - 4); j += 4) {
	    /* loop over values within the row */
	    m = 0;
	    for (l = 0; l < DIM; l++) {
		if (rowTypes[(i*DIM)+l]) {
		    /* there is a value here */
		    H1vals[((k * DIM) + (m*4)) + 0] = H1[(l*numRows)+rowPerm[k]];
		    H1vals[((k * DIM) + (m*4)) + 1] = H1[(l*numRows)+rowPerm[k+1]];
		    H1vals[((k * DIM) + (m*4)) + 2] = H1[(l*numRows)+rowPerm[k+2]];
		    H1vals[((k * DIM) + (m*4)) + 3] = H1[(l*numRows)+rowPerm[k+3]];
		    m++;
		}
	    }
	    k += 4;
	}
	for (; j < rowTypeCounts[i]; j++) {
	    /* handle stray rows */
	    m = 0;
	    for (l = 0; l < DIM; l++) {
		if (rowTypes[(i*DIM)+l]) {
		    //if (H1[(l*numRows) + k] != 0.0) {
		    H1vals[(k * DIM) + m] = H1[(l*numRows)+rowPerm[k]];
		    m++;
		}
	    }
	    k++;
	}
    }
    if (k != numRows) {
	// sanity check failed
	logMessage(5, "Internal error: wrong number of rows (%d) in modal plate startup code!", k);
	exit(1);
    }

    // compile the required information for each row type
    rowInfo = new RowInfo[numRowTypes];
    j = 0;

    for (i = 0; i < numRowTypes; i++) {
	rowInfo[i].start = j;
	rowInfo[i].count = rowTypeCounts[i];

	// compute nnz for this row type
	rowInfo[i].nnz = 0;
	for (k = 0; k < DIM; k++) {
	    if (rowTypes[(i*DIM)+k] != 0) rowInfo[i].nnz++;
	}

	// generate q1 index for this row type
	rowInfo[i].q1index = new int[rowInfo[i].nnz];
	l = 0;
	for (k = 0; k < DIM; k++) {
	    if (rowTypes[(i*DIM)+k] != 0) {
		rowInfo[i].q1index[l] = k;
		l++;
	    }
	}

	j += rowTypeCounts[i];
    }

    delete[] rowTypes;
}

void ModalPlate::instr_def(double fsd)
{
    double xi;
    int m;
    double *c = new double[DIM];

    ov = omega_calculator();

    /* damping ratios */
    for (m = 0; m < DIM; m++) {
	xi = 6.0 * log(10.0) / (loss1*pow((ov[m*3] / ov[0]), 0.5) + loss2);
	c[m] = 6.0 * log(10.0) / xi;
    }

    for (m = 0; m < DIM; m++) {
	C[m] = (fsd*fsd + c[m]*fsd/2.0);
	C1[m] = (-2.0*fsd*fsd + ov[m*3]*ov[m*3]);
	C2[m] = (fsd*fsd - c[m]*fsd/2.0);
	
	C1[m] = C1[m] / C[m];
	C2[m] = C2[m] / C[m];
	Cnorm[m] = C[m]; /* Cnorm is just a copy of C */
    }
    delete[] c;
}

static int omega_compare(const void *z1, const void *z2)
{
    double *zazi1 = (double *)z1;
    double *zazi2 = (double *)z2;
    if (zazi1[0] < zazi2[0]) return -1;
    if (zazi1[0] > zazi2[0]) return 1;

    if (zazi1[1] < zazi2[1]) return -1;
    if (zazi1[1] > zazi2[1]) return 1;

    if (zazi1[2] < zazi2[2]) return -1;
    if (zazi1[2] > zazi2[2]) return 1;
    return 0;
}

double *ModalPlate::omega_calculator()
{
    double D;
    int ind, m, n;
    double *zazi;
    double gf;

    D = Young * (h*h*h) / 12.0 / (1.0 - (nu*nu));

    zazi = new double[DD * DD * 3];

    ind = 0;
    for (m = 1; m <= DD; m++) {
	for (n = 1; n <= DD; n++) {
	    gf = sqrt(D/rho/h) * ((((double)m)*M_PI/Lx)*(((double)m)*M_PI/Lx) +
				  (((double)n)*M_PI/Ly)*(((double)n)*M_PI/Ly));
	    zazi[(ind*3)+0] = gf;
	    zazi[(ind*3)+1] = (double)m;
	    zazi[(ind*3)+2] = (double)n;
	    ind++;
	}
    }

    qsort(zazi, DD*DD, 3*sizeof(double), omega_compare);

    return zazi;
}

#define PI2 (M_PI*M_PI)
#define PI3 (M_PI*M_PI*M_PI)
#define PI4 (M_PI*M_PI*M_PI*M_PI)
#define PI5 (M_PI*M_PI*M_PI*M_PI*M_PI)

void ModalPlate::Hcalc_unstable(double *coeff1, int DIM, int A, double Lx,
				double Ly, int S, double *H1)
{
    int m, n, p;
    int n1, p1, n2, p2;
    double *mthing;

    double tmp;
    double m1, m2, m3, m4, m5, m6;
    double fac, rowmax;

    double *y;

    /* allocate temporary storage */
    mthing = new double[S*DIM*DIM];
    y = new double[DIM*DIM*4];

    /* generate y */
    i_phi(DIM, Lx, Ly, y);

    /* generate mthing (column major) */
    for (m = 0; m < S; m++) {
	int ma = m / A; /* needed for g1, g2, g5 */
	int mb = m % A; /* needed for g3, g4, g6 */
	for (n = 0; n < DIM; n++) {
	    n1 = y[(n*4)+1];
	    n2 = y[(n*4)+2];
	    for (p = 0; p < DIM; p++) {
		p1 = y[(p*4)+1];
		p2 = y[(p*4)+2];

		tmp = (i1_mat(A,DIM,Lx,ma+1,n1,p1) + i2_mat(A,DIM,Lx,ma+1,n1,p1) +
		       i3_mat(A,DIM,Lx,ma+1,n1,p1) + i4_mat(A,DIM,Lx,ma+1,n1,p1) +
		       i5_mat(A,DIM,Lx,ma+1,n1,p1));
		m1 = tmp * ((double)(n1*n1));
		m2 = tmp * ((double)(p1*p1));

		m5 = (i9_mat (A,DIM,Lx,ma+1,n1,p1) + i10_mat(A,DIM,Lx,ma+1,n1,p1) +
		      i11_mat(A,DIM,Lx,ma+1,n1,p1) + i12_mat(A,DIM,Lx,ma+1,n1,p1) +
		      i13_mat(A,DIM,Lx,ma+1,n1,p1)) * (double)p1 * (double)n1;

		tmp = (i1_mat(A,DIM,Ly,mb+1,n2,p2) + i2_mat(A,DIM,Ly,mb+1,n2,p2) +
		       i3_mat(A,DIM,Ly,mb+1,n2,p2) + i4_mat(A,DIM,Ly,mb+1,n2,p2) +
		       i5_mat(A,DIM,Ly,mb+1,n2,p2));
		m3 = tmp * ((double)(n2*n2));
		m4 = tmp * ((double)(p2*p2));

		m6 = (i9_mat (A,DIM,Ly,mb+1,n2,p2) + i10_mat(A,DIM,Ly,mb+1,n2,p2) +
		      i11_mat(A,DIM,Ly,mb+1,n2,p2) + i12_mat(A,DIM,Ly,mb+1,n2,p2) +
		      i13_mat(A,DIM,Ly,mb+1,n2,p2)) * (double)p2 * (double)n2;

		mthing[m + (n*S) + (p*DIM*S)] = m1*m4 + m2*m3 - 2.0*m5*m6;
	    }
	}
    }

    fac = 4.0 * PI4 / (Lx*Lx*Lx) / (Ly*Ly*Ly);

    /* generate next row of H1 */
    for (n = 0; n < S; n++) {
	rowmax = 0.0;

	/*
	 * this row is product of coeff1 column n, with:
	 *   (m1.*m4 + m2.*m3 - 2*m5.*m6): mthing
	 */
	for (p = 0; p < (DIM*DIM); p++) {
	    tmp = 0.0;
	    for (m = 0; m < S; m++) {
		tmp += coeff1[(n*S)+m] * mthing[(p*S)+m];
	    }

	    /* apply a scalar factor */
	    tmp *= fac;

	    /* store in H1, in column-major order */
	    H1[(p*S)+n] = tmp;

	    /* find the row max */
	    if (fabs(tmp) > rowmax) rowmax = fabs(tmp);
	}

	/* now zero all values that are 10 orders of magnitude lower than the row max */
	for (p = 0; p < (DIM*DIM); p++) {
	    if ((fabs(H1[(p*S)+n]) / rowmax) < 1e-10) H1[(p*S)+n] = 0.0;
	}
    }

    delete[] mthing;
    delete[] y;
}


static int i_phi_compare(const void *i1, const void *i2)
{
    double *i_phi1 = (double *)i1;
    double *i_phi2 = (double *)i2;
    if (i_phi1[0] < i_phi2[0]) return -1;
    if (i_phi1[0] > i_phi2[0]) return 1;

    if (i_phi1[1] < i_phi2[1]) return -1;
    if (i_phi1[1] > i_phi2[1]) return 1;

    if (i_phi1[2] < i_phi2[2]) return -1;
    if (i_phi1[2] > i_phi2[2]) return 1;

    if (i_phi1[3] < i_phi2[3]) return -1;
    if (i_phi1[3] > i_phi2[3]) return 1;
    return 0;
}

/* y should be big enough to hold a (DIM*DIM)x4 array, row major */
void ModalPlate::i_phi(int DIM, double Lx, double Ly, double *y)
{
    int m, n, ii;
    double h;

    ii = 0;
    for (m = 1; m <= DIM; m++) {
	for (n = 1; n <= DIM; n++) {
	    if ((m & 1) == 0) {
		if ((n & 1) == 0) {
		    h = 4.0;
		}
		else {
		    h = 3.0;
		}
	    }
	    else {
		if ((n & 1) == 0) {
		    h = 2.0;
		}
		else {
		    h = 1.0;
		}
	    }

	    y[(ii*4)+0] = (((double)m)*M_PI/Lx)*(((double)m)*M_PI/Lx) +
		(((double)n)*M_PI/Ly)*(((double)n)*M_PI/Ly);
	    y[(ii*4)+1] = (double)m;
	    y[(ii*4)+2] = (double)n;
	    y[(ii*4)+3] = h;
	    ii++;
	}
    }
    qsort(y, DIM*DIM, 4*sizeof(double), i_phi_compare);
}

/* m, n, p are one-based Matlab indices */
double ModalPlate::i1_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    if ((m1 == 0) && (n == p)) {
	return L/2.0;
    }
    else if ((m1 == (p-n)) || (m1 == (n-p))) {
	return L/4.0;
    }
    else if ((m1 == (-n-p)) || (m1 == (n+p))) {
	return -L/4.0;
    }
    return 0.0;
}

double ModalPlate::i2_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    double L4 = L*L*L*L;
    double L5 = L4*L;
    double dp = (double)p;
    double p3 = p*p*p;
    double p5 = p3*p*p;
    double npp = (double)n + (double)p;
    double npp4 = npp*npp*npp*npp;
    double npp5 = npp4*npp;
    double nmp = (double)n - (double)p;
    double nmp4 = nmp*nmp*nmp*nmp;
    double nmp5 = nmp4*nmp;
    double m1pm1 = 1.0;
    if (m1 & 1) m1pm1 = -1.0;

    if (n == p) {
	return (+15.0/L4*(m1pm1 + 1.0)) * (L5*(4.0*PI5*p5 - 20.0*PI3*p3 + 30.0*M_PI*dp)) /
	    (40.0*PI5*p5);
    }
    return -(+15.0/L4*(m1pm1 + 1.0)) * (8796093022208.0*L*((sin(M_PI*npp)*((1713638851887625.0*L4*npp4)/17592186044416.0 - (8334140006820045.0*L4*npp*npp)/70368744177664.0 + 24.0*L4) + 4.0*M_PI*L*L*cos(M_PI*npp)*npp*((2778046668940015.0*L*L*npp*npp)/281474976710656.0 - 6.0*L*L))/npp5 - (sin(M_PI*nmp)*((1713638851887625.0*L4*nmp4)/17592186044416.0 - (8334140006820045.0*L4*nmp*nmp)/70368744177664.0 + 24.0*L4) + 4.0*M_PI*L*L*cos(M_PI*nmp)*nmp*((2778046668940015.0*L*L*nmp*nmp)/281474976710656.0 - 6.0*L*L))/nmp5))/5383555227996211.0;
}

double ModalPlate::i3_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    double L3 = L*L*L;
    double L4 = L3*L;
    double dp = (double)p;
    double p4 = dp*dp*dp*dp;
    double nmp = (double)n - (double)p;
    double nmp4 = nmp*nmp*nmp*nmp;
    double npp = (double)n + (double)p;
    double npp4 = npp*npp*npp*npp;
    double m1pm1 = 1.0;
    if (m1 & 1) m1pm1 = -1.0;

    if (n == p) {
	return -(-4.0/L3*(7.0*m1pm1 + 8.0))*(L4*(6.0*PI2*dp*dp - 2.0*PI4*p4))/(16.0*PI4*p4);
    }
    return (-4.0/L3*(7.0*m1pm1 + 8.0))*(L*((6.0*L3)/nmp4 - (6.0*L3)/npp4))/(2.0*PI4) + (-4.0/L3*(7.0*m1pm1 + 8.0))*(L*(3.0*L*cos(M_PI*npp)*(2.0*L*L - L*L*PI2*npp*npp)/npp4 - (3.0*L*cos(M_PI*nmp)*(2.0*L*L - L*L*PI2*nmp*nmp))/nmp4))/(2.0*PI4);
}

double ModalPlate::i4_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    double L3 = L*L*L;
    double dp = (double)p;
    double p3 = dp*dp*dp;
    double nmp = (double)n - (double)p;
    double npp = (double)n + (double)p;
    double m1pm1 = 1.0;
    if (m1 & 1) m1pm1 = -1.0;

    if (n == p) {
	return -(6.0/(L*L)*(2.0*m1pm1 + 3.0)) * (L3*(6.0*M_PI*dp - 4.0*PI3*p3)) / (24.0*PI3*p3);
    }
    return (6/(L*L)*(2.0*m1pm1 + 3.0)) * (L3*cos(M_PI*nmp)) / (PI2*nmp*nmp) - (6.0/(L*L)*(2.0*m1pm1 + 3.0)) * (L3*cos(M_PI*npp)) / (PI2*npp*npp);
}

double ModalPlate::i5_mat(int A, int DIM, double L, int m, int n, int p)
{
    if (n == p) return -L/2.0;
    return 0.0;
}

double ModalPlate::i9_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    if ((m1 == 0) && (n == p)) return L/2.0;
    else if ((m1 == (p-n)) || (m1 == (n-p)) || (m1 == (-n-p)) || (m1 == (n+p))) return L/4.0;
    return 0.0;
}

double ModalPlate::i10_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    double L4 = L*L*L*L;
    double L5 = L4*L;
    double dn = (double)n;
    double n3 = dn*dn*dn;
    double n5 = n3*dn*dn;
    double npp = (double)n + (double)p;
    double npp2 = npp*npp;
    double npp4 = npp2*npp2;
    double nmp = (double)n - (double)p;
    double nmp2 = nmp*nmp;
    double nmp4 = nmp2*nmp2;
    double m1pm1 = 1.0;
    if (m1 & 1) m1pm1 = -1.0;

    if (n == p) {
	return (15.0/L4*(m1pm1 + 1.0)) * (L5*(4.0*PI5*n5 + 20.0*PI3*n3 - 30.0*M_PI*dn)) / (40.0*PI5*n5);
    }
    return -(15.0/L4*(m1pm1 + 1.0)) * (L*((4.0*M_PI*L*L*cos(M_PI*npp)*(6.0*L*L - L*L*PI2*npp2))/npp4 + (4.0*M_PI*L*L*cos(M_PI*nmp)*(6.0*L*L - L*L*PI2*nmp2))/nmp4))/(2.0*PI5);
}

double ModalPlate::i11_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    double L3 = L*L*L;
    double L4 = L3*L;
    double dp = (double)p;
    double p2 = dp*dp;
    double nmp = (double)n - (double)p;
    double nmp2 = nmp*nmp;
    double nmp4 = nmp2*nmp2;
    double npp = (double)n + (double)p;
    double npp2 = npp*npp;
    double npp4 = npp2*npp2;
    double m1pm1 = 1.0;
    if (m1 & 1) m1pm1 = -1.0;

    if (n == p) {
	return (-4.0/L3*(7.0*m1pm1 + 8.0))*L4/8.0 + (-4.0/L3*(7.0*m1pm1 + 8.0))*(3.0*L4)/(8.0*PI2*p2);
    }
    return (-4.0/L3*(7.0*m1pm1 + 8.0))*(L*((6.0*L3)/nmp4 + (6.0*L3)/npp4))/(2.0*PI4) - (-4.0/L3*(7.0*m1pm1 + 8.0))*(L*((3.0*L*cos(M_PI*npp)*(2.0*L*L - L*L*PI2*npp2))/npp4 + (3.0*L*cos(M_PI*nmp)*(2.0*L*L - L*L*PI2*nmp2))/nmp4))/(2.0*PI4);
}

double ModalPlate::i12_mat(int A, int DIM, double L, int m, int n, int p)
{
    int m1 = m - 1;
    double L3 = L*L*L;
    double dp = (double)p;
    double p2 = dp*dp;
    double npp = (double)n + (double)p;
    double npp2 = npp*npp;
    double nmp = (double)n - (double)p;
    double nmp2 = nmp*nmp;
    double m1pm1 = 1.0;
    if (m1 & 1) m1pm1 = -1.0;

    if (n == p) {
	return (6.0/(L*L)*(2.0*m1pm1 + 3.0))*L3/6.0 + (6.0/(L*L)*(2.0*m1pm1 + 3.0))*L3/(4.0*PI2*p2);
    }
    return (6.0/(L*L)*(2.0*m1pm1 + 3.0))*L3*cos(M_PI*nmp)/(PI2*nmp2) + (6.0/(L*L)*(2.0*m1pm1 + 3.0))*L3*cos(M_PI*npp)/(PI2*npp2);
}

double ModalPlate::i13_mat(int A, int DIM, double L, int m, int n, int p)
{
    if (n == p) return -L/2.0;
    return 0.0;
}

int ModalPlate::eigencalcUnstable(int A, double Lx, double Ly, double *coeff1)
{
    int m, n, p, q, r, s;
    int Asq = A*A;
    int S;
    int bestidx;

    double *K, *M;
    double *L, *U, *Linv, *Uinv, *C, *tmp;
    double *d, *v;
    double *VEC, *VAL;
    double bestval = 0.0;
    double norm;

    double *NN, *MM, *nmatr, *nmatr2;

    /* allocate all temporary storage needed */
    K = new double[Asq*Asq];
    M = new double[Asq*Asq];
    L = new double[Asq*Asq];
    U = new double[Asq*Asq];
    Linv = new double[Asq*Asq];
    Uinv = new double[Asq*Asq];
    C = new double[Asq*Asq];
    tmp = new double[Asq*Asq];
    d = new double[Asq];
    v = new double[Asq*Asq];
    VAL = new double[Asq];
    VEC = new double[Asq*Asq];

    /* calculate K and M matrices */
    r = 0;
    for (m = 0; m < A; m++) {
	for (n = 0; n < A; n++) {
	    s = 0;
	    for (p = 0; p < A; p++) {
		for (q = 0; q < A; q++) {
		    K[(r*Asq)+s] = int1(m,p,Lx)*int2(n,q,Ly) + int2(m,p,Lx)*int1(n,q,Ly) +
			2.0*int4(m,p,Lx)*int4(n,q,Ly);
		    M[(r*Asq)+s] = int2(m,p,Lx)*int2(n,q,Ly);
		    s++;
		}
	    }
	    r++;
	}
    }

    /* solve generalised Eigenvalue problem for K and M */
    /* get Cholesky decomposition of M */
    denseCholeskyDecomp(A*A, M, L, U);

    /* get inversion of lower triangle */
    invertLowerTriangle(A*A, L, Linv);

    /* get transpose of inversion */
    transposeDenseMatrix(A*A, Linv, Uinv);

    /* tmp = Linv*K */
    denseMatrixMatrixMultiply(A*A, Linv, K, tmp);

    /* C = Linv*K*Uinv */
    denseMatrixMatrixMultiply(A*A, tmp, Uinv, C);

    /* compute eigenvalues of C */
    getEigenvalues(A*A, C, d, v);


    /* sort eigenvalues into order, remove negative or zero ones. also sort eigenvectors */
    S = 0;
    do {
	bestval = 1e40;
	bestidx = -1;
	for (m = 0; m < Asq; m++) {
	    if ((d[m] < bestval) && (d[m] > 0.0)) {
		bestval = d[m];
		bestidx = m;
	    }
	}
	
	if (bestidx >= 0) {
	    /* found one */
	    VAL[S] = bestval;
	    d[bestidx] = 0.0;

	    /* copy the vector as well */
	    for (n = 0; n < Asq; n++) {
		tmp[n] = v[(n*Asq) + bestidx];
	    }

	    /* multiply the vector by Uinv to correct it */
	    denseMatrixVectorMultiply(Linv, tmp, &VEC[S*Asq], Asq, Asq);

	    /* normalise the vector */
	    norm = 0.0;
	    for (n = 0; n < Asq; n++) {
		norm += (VEC[S*Asq+n]*VEC[S*Asq+n]);
	    }
	    norm = sqrt(norm);
	    for (n = 0; n < Asq; n++) {
		VEC[S*Asq+n] /= norm;
	    }

	    S++;
	}
    } while (bestidx >= 0);

    /* create the two integral matrices */
    NN = int2_mat(A, Lx); /* these are both symmetric */
    MM = int2_mat(A, Ly);

    /* NN is a single column, then replicated Asq times */
    /* MM is a single row, then replicated Asq times */
    /* element-wise multiply them to get nmatr in *column-major order* */
    nmatr = new double[Asq*Asq];
    for (m = 0; m < Asq; m++) {
	for (n = 0; n < Asq; n++) {
	    nmatr[(n*Asq)+m] = NN[m] * MM[n];
	}
    }
    
    /* now we treat nmatr as a 4D AxAxAxA array and permute the dimensions into the order:
     * 4 1 3 2 */
    nmatr2 = new double[Asq*Asq];
    for (m = 0; m < A; m++) {
	for (n = 0; n < A; n++) {
	    for (p = 0; p < A; p++) {
		for (q = 0; q < A; q++) {
		    nmatr2[q + (m*A) + (p*Asq) + (n*Asq*A)] =
			nmatr[m + (n*A) + (p*Asq) + (q*Asq*A)];
		}
	    }
	}
    }

    /* from here on nmatr(2) is treated as a row vector */

    /* loop over eigenvectors */
    for (m = 0; m < S; m++) {
	double svm = sqrt(VAL[m]);
	double temp, temp2, temp3, norm, snorm;

	norm = 0.0;
	for (p = 0; p < Asq; p++) { /* columns of temp3 */
	    for (q = 0; q < Asq; q++) { /* rows of temp3 */
		temp = VEC[(m*Asq)+q];
		temp2 = VEC[(m*Asq)+p];
		temp3 = temp * temp2;
		norm += temp3 * nmatr2[(p*Asq)+q];
	    }
	}

	snorm = sqrt(norm);
	for (n = 0; n < Asq; n++) {
	    tmp[(m*Asq)+n] = VEC[(m*Asq)+n] / snorm / svm;
	}
    }

    /* compute actual S value */
    S = S / 2;

    /* copy result into actual buffer */
    for (m = 0; m < S; m++) {
	for (n = 0; n < S; n++) {
	    coeff1[(n*S)+m] = tmp[(n*Asq)+m];
	}
    }

    delete[] K;
    delete[] M;
    delete[] L;
    delete[] U;
    delete[] Linv;
    delete[] Uinv;
    delete[] C;
    delete[] tmp;
    delete[] d;
    delete[] v;
    delete[] VAL;
    delete[] VEC;
    delete[] NN;
    delete[] MM;
    delete[] nmatr;
    delete[] nmatr2;

    return S;
}


/* integration functions */
double ModalPlate::int1(int m, int p, double L)
{
    double y;
    if ((m == 0) && (p == 0)) {
	y = 720.0 / (L*L*L);
    }
    else if (m == p) {
	double m1pm = 1.0;
	if (m & 1) m1pm = -1.0; /* (-1)^m */

	y = (PI4 * ((double)m*m*m*m) - 672.0*m1pm - 768.0) / (2.0*L*L*L);
    }
    else if ((m == 0) || (p == 0)) {
	y = 0.0;
    }
    else {
	double m1pm = 1.0;
	double m1pp = 1.0;
	if (m & 1) m1pm = -1.0; /* (-1)^m */
	if (p & 1) m1pp = -1.0; /* (-1)^p */

	y = -(24.0*(7.0*m1pm + 7.0*m1pp + 8.0*m1pm*m1pp + 8.0)) / (L*L*L);
    }
    return y;
}

double ModalPlate::int2(int m, int p, double L)
{
    double y;
    double m4, p4;
    double m1pm = 1.0;
    double m1pp = 1.0;
    if (m & 1) m1pm = -1.0; /* (-1)^m */
    if (p & 1) m1pp = -1.0; /* (-1)^p */

    m4 = (double)m;
    m4 = m4*m4*m4*m4;
    p4 = (double)p;
    p4 = p4*p4*p4*p4;

    if ((m == 0) && (p == 0)) {
	y = (10.0*L)/7.0;
    }
    else if (m == p) {
	y = (67.0*L)/70.0 - (m1pm*L)/35.0 - (768.0*L)/(PI4*m4) - (672.0*m1pm*L)/(PI4*m4);
    }
    else if (m == 0) {
	y = (3.0*L*(m1pp+1.0)*(PI4*p4 - 1680.0)) / (14.0*PI4*p4);
    }
    else if (p == 0) {
	y = (3.0*L*(m1pm+1.0)*(PI4*m4 - 1680.0)) / (14.0*PI4*m4);
    }
    else {
	y = -(L*(11760.0*m1pm + 11760.0*m1pp - 16.0*PI4*m4 + 13440.0*m1pm*m1pp +
		 m1pm*PI4*m4 + m1pp*PI4*m4 - 16.0*m1pm*m1pp*PI4*m4 + 13440.0)) / (70.0*PI4*m4) -
	    (L*(13440.0*m4 + 11760.0*m1pm*m4 + 11760.0*m1pp*m4 + 13440.0*m1pm*m1pp*m4)) / (70.0*PI4*m4*p4);
    }
    return y;
}

double ModalPlate::int4(int m, int p, double L)
{
    double y;
    double m2 = (double)(m*m);
    double p2 = (double)(p*p);
    double m1pm = 1.0;
    double m1pp = 1.0;
    if (m & 1) m1pm = -1.0; /* (-1)^m */
    if (p & 1) m1pp = -1.0; /* (-1)^p */

    if ((m == 0) && (p == 0)) {
	y = 120.0 / (7.0*L);
    }
    else if ((m == p) && (m != 0)) {
	y = (768.0*PI2*m2 - 47040.0*m1pm + 35.0*PI4*m2*m2 + 432.0*m1pm*PI2*m2 - 53760.0) /
	    (70.0*L*PI2*m2);
    }
    else if (m == 0) {
	y = (60.0*(m1pp + 1.0)*(PI2*p2 - 42.0)) / (7.0*L*PI2*p2);
    }
    else if (p == 0) {
	y = (60.0*(m1pm + 1.0)*(PI2*m2 - 42.0)) / (7.0*L*PI2*m2);
    }
    else {
	y = 192.0/35.0/L*(1.0 + m1pm*m1pp) - 192.0/m2/p2/L/PI2*((p2+m2)*(1.0+m1pm*m1pp)) -
	    168.0/m2/p2/L/PI2*((p2+m2)*(m1pm+m1pp)) + 108.0/35.0/L*(m1pm+m1pp);
    }
    return y;
}

double *ModalPlate::int2_mat(int tt, double L)
{
    double *y;
    int m, p;
    double m1pm, m1pp, m4;
    double dp;

    y = new double[tt*tt];

    for (m = 0; m < tt; m++) {
	m4 = (double)m;
	m4 = m4*m4*m4*m4;
	m1pm = 1.0;
	if (m & 1) m1pm = -1.0;

	for (p = 1; p < tt; p++) {
	    dp = (double)p;
	    m1pp = 1.0;
	    if (p & 1) m1pp = -1.0;

	    if (m == 0) {
		/* top row */
		y[(m*tt)+p] = (3.0*L*(m1pp + 1.0) * (PI4*(dp*dp*dp*dp) - 1680.0)) / (14.0*PI4*(dp*dp*dp*dp));
	    }
	    else {
		y[(m*tt)+p] = -(L*(11760.0*m1pm + 11760.0*m1pp - 16.0*PI4*m4 +
				   13440.0*m1pm*m1pp + m1pm*PI4*m4 + m1pp*PI4*m4 -
				   16.0*m1pp*PI4*(m4*m1pm) + 13440.0))/(70.0*PI4*m4) -
		    (L*(13440.0*m4 + 11760.0*m1pm*m4 + 11760*m1pp*m4 + 13440.0*m1pm*m1pp*m4)) /
		    (70.0*PI4*m4*dp*dp*dp*dp);
	    }
	}

	/* diagonal element */
	y[(m*tt)+m] = (67.0*L)/70.0 - (m1pm*L)/35.0 - (768.0*L)/(PI4*m4) -
	    (672.0*m1pm*L)/(PI4*m4);

	/* left hand side */
	y[(m*tt)] = (3.0*L*(m1pm + 1.0) * (PI4*m4 - 1680.0)) / (14.0*PI4*m4);
    }

    /* top left */
    y[0] = (10.0*L)/7.0;

    return y;
}


int ModalPlate::getIndexf(double x, double y, double z)
{
    logMessage(5, "ERROR: getIndexf called on modal plate!");
    exit(1);
}

int ModalPlate::getIndex(int x, int y, int z)
{
    logMessage(5, "ERROR: getIndex called on modal plate!");
    exit(1);
}

void ModalPlate::getInterpolationInfo(InterpolationInfo *info, double x, double y, double z)
{
    logMessage(5, "ERROR: getInterpolationInfo called on modal plate!");
    exit(1);
}

void ModalPlate::runTimestep(int n)
{
    int i, j, k;

#ifdef USE_GPU
    if (gpuModalPlate != NULL) {
	gpuModalPlate->runTimestep(n, u, u1, u2);
	runInputs(n, u, u1, u2);
	return;
    }
#endif

#ifdef USE_AVX
    if (H1vals) {
	// run optimised AVX version of t1 = (H1*q1)
	k = 0;
	for (i = 0; i < numRowTypes; i++) {
	    // loop over rows of this type in 4s
	    for (j = 0; j < (rowTypeCounts[i]-4); j += 4) {
		avxUpdate4Rows(u1, &H1vals[k*DIM], &t1perm[k], rowInfo[i].q1index,
			       rowInfo[i].nnz);
		k += 4;
	    }
	    // do odd rows at the end
	    for (; j < rowTypeCounts[i]; j++) {
		avxUpdateSingleRow(u1, &H1vals[k*DIM], &t1perm[k], rowInfo[i].q1index,
				   rowInfo[i].nnz);
		k++;
	    }
	}

	// permute t1 into standard order for rest of loop
	for (i = 0; i < (A*DIM); i++) {
	    t1[i] = t1perm[reversePerm[i]];
	}
    }
    else {
#endif
	// run standard version
	/* t1 = (H1*q1); */
	denseMatrixVectorMultiply(H1, u1, t1, (A*DIM), DIM);
#ifdef USE_AVX
    }
#endif

    /* t2 = t1*q1; */
    denseMatrixVectorMultiply(t1, u1, t2, A, DIM);

    /* G = t1.'*t2; */
    denseMatrixVectorMultiplyTransposed(t1, t2, G, A, DIM);

    /* G = G./C; */
    for (i = 0; i < DIM; i++) {
	G[i] = G[i] / C[i];
    }

    /* q = - C1.*q1 - C2.*q2 - G + inputs; */
    for (i = 0; i < DIM; i++) {
	u[i] = -(C1[i]*u1[i]) - (C2[i]*u2[i]) - G[i];
    }

    runInputs(n, u, u1, u2);
}

// parallel function to update one row type (AVX accelerated)
void ModalPlate::runRowType(int n, int t)
{
#ifdef USE_AVX
    int j, k;

    k = rowInfo[t].start;

    // loop over rows of this type in 4s
    for (j = 0; j < (rowTypeCounts[t]-4); j += 4) {
	avxUpdate4Rows(u1, &H1vals[k*DIM], &t1perm[k], rowInfo[t].q1index,
		       rowInfo[t].nnz);
	k += 4;
    }
    // do odd rows at the end
    for (; j < rowTypeCounts[t]; j++) {
	avxUpdateSingleRow(u1, &H1vals[k*DIM], &t1perm[k], rowInfo[t].q1index,
			   rowInfo[t].nnz);
	k++;
    }
#else
    logMessage(5, "Internal error: ModalPlate::runRowType called in non-AVX build!");
    exit(1);
#endif
}

// serial function to finish timestep update after parallel updates of row types
// have all completed
void ModalPlate::finishUpdate(int n)
{
    int i;

    // permute t1 into standard order for rest of loop
    for (i = 0; i < (A*DIM); i++) {
	t1[i] = t1perm[reversePerm[i]];
    }

    /* t2 = t1*q1; */
    denseMatrixVectorMultiply(t1, u1, t2, A, DIM);

    /* G = t1.'*t2; */
    denseMatrixVectorMultiplyTransposed(t1, t2, G, A, DIM);

    /* G = G./C; */
    for (i = 0; i < DIM; i++) {
	G[i] = G[i] / C[i];
    }

    /* q = - C1.*q1 - C2.*q2 - G + inputs; */
    for (i = 0; i < DIM; i++) {
	u[i] = -(C1[i]*u1[i]) - (C2[i]*u2[i]) - G[i];
    }

    runInputs(n, u, u1, u2);
}


void ModalPlate::getParallelTasks(vector<Task*> &tasks)
{
    // if using AVX version, each row type is a separate parallel task
    // otherwise, the whole component is a single paralle task
    if (H1vals) {
	int i;
	for (i = 0; i < numRowTypes; i++) {
	    tasks.push_back(new TaskModalPlateRow(this, i));
	}
    }
    else {
	tasks.push_back(new TaskWholeComponent(this));
    }
}

void ModalPlate::getSerialTasks(vector<Task*> &tasks)
{
    // if using AVX version, the end part of the main update is a serial
    // task. otherwise, no serial tasks
    if (H1vals) {
	tasks.push_back(new TaskModalPlateEnd(this));
    }
}


int ModalPlate::getGPUScore()
{
    if (SettingsManager::getInstance()->getBoolSetting(name, "disable_gpu")) return GPU_SCORE_NO;

    /*if ((DIM >= 200) && (DIM < 2500)) {
	return GPU_SCORE_GOOD;
	}*/
    return GPU_SCORE_NO;
}

int ModalPlate::getGPUMemRequired()
{
    return ((DIM*7) + (A) + (A*DIM) + (DIM*DIM*A)) * sizeof(double);
}

bool ModalPlate::moveToGPU()
{
#ifdef USE_GPU
    // try to move to GPU
    double *d_u, *d_u1, *d_u2;
    int i;
    gpuModalPlate = new GPUModalPlate(A, DIM, H1, C, C1, C2, &d_u, &d_u1, &d_u2);
    if (!gpuModalPlate->isOK()) {
	// didn't work, keep on host
	logMessage(1, "Moving modal plate %s to GPU failed", name.c_str());
	delete gpuModalPlate;
	gpuModalPlate = NULL;
	return false;
    }

    // replace state arrays
    delete[] u;
    delete[] u1;
    delete[] u2;
    u = d_u;
    u1 = d_u1;
    u2 = d_u2;

    // move inputs to GPU as well
    for (i = 0; i < inputs.size(); i++) {
	inputs[i]->moveToGPU();
    }
    logMessage(1, "Modal plate %s moved to GPU", name.c_str());
    return true;
#else
    return false;
#endif
}
