/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

/*
 * There are a number of differences between the Matlab version and this one.
 * Matlab takes outputs from the vertical (w) polarisation as well as the horizontal
 * (u). Being able to do this would require quite a few changes to the framework, so
 * this version doesn't allow outputs from w. They probably wouldn't be useful for
 * anything other than debugging anyway as the vertical position typically changes
 * too slowly to be audible, and the interesting sounds are in the horizontal.
 *
 * The contact Newton solver is vector, so it should cope with bows and fingers
 * that overlap or are very close together. The friction solver is scalar so it
 * might not give quite the right answer in those circumstances. The interpolation
 * and spreading matrices (J* and I*) aren't matrices in this version; instead, a
 * base index and 4 co-efficients are stored, since that's all that's needed. It
 * makes some of the operations a bit more verbose but it should be faster. The
 * co-efficients aren't divided by h like they are in the Matlab.
 *
 * Another potential source of confusion is that bows and fingers are both
 * represented by the Bow structure, which has an "isFinger" flag to tell them
 * apart. The bow and finger update schemes are similar enough for this to make
 * sense.
 */

#include "BowedString.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "Input.h"
#include "MathUtil.h"

#include "WavWriter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
using namespace std;

#define MAX_OBJS 8

/*
 * String parameters for the preset instruments
 */
struct StringParameters {
    double f0;
    double diameter;
    double rho;
    double E;
};

static const StringParameters violinStrings[5][4] = {
    {
	{ 659.3, 3.3e-4, 4.1e-4, 6.25e10 },
	{ 440.0, 6.0e-4, 7.2e-4, 1.95e10 },
	{ 293.7, 8.8e-4, 0.0016, 4.56e9 },
	{ 196.0, 8.5e-4, 0.0028, 4.79e9 }
    },
    {
	{ 659.3, 2.9e-4, 5.0e-4, 1.34e11 },
	{ 440.0, 6.1e-4, 8.8e-4, 2.22e10 },
	{ 293.7, 7.8e-4, 9.9e-4, 6.87e9 },
	{ 196.0, 7.8e-4, 0.0032, 5.51e9 }
    },
    {
	{ 659.3, 2.8e-4, 5.7e-4, 1.86e11 },
	{ 440.0, 7.5e-4, 5.8e-4, 4.92e9 },
	{ 293.7, 7.4e-4, 0.0014, 1.29e10 },
	{ 196.0, 6.5e-4, 0.0037, 1.9e10 }
    },
    {
	{ 659.3, 3.2e-4, 4.1e-4, 7.51e10 },
	{ 440.0, 6.3e-4, 8.6e-4, 2.31e10 },
	{ 293.7, 6.2e-4, 9.7e-4, 1.83e10 },
	{ 196.0, 9.8e-4, 0.0036, 2.84e9 }
    },
    {
	{ 659.3, 2.4e-4, 3.6e-4, 1.86e11 },
	{ 440.0, 6.5e-4, 8.6e-4, 1.84e10 },
	{ 293.7, 7.8e-4, 0.0018, 1.63e10 },
	{ 196.0, 7.3e-4, 0.0019, 8.91e9 }
    }
};

static const StringParameters violaStrings[2][4] = {
    {
	{ 440.0, 3.6e-4, 8.2e-4, 8.13e10 },
	{ 293.7, 4.4e-4, 0.00126, 5.53e10 },
	{ 196.0, 6.7e-4, 0.00217, 8.01e9 },
	{ 130.8, 7.2e-4, 0.00487, 1.81e10 }
    },
    {
	{ 440.0, 3.5e-4, 5.8e-4, 1.14e11 },
	{ 293.7, 3.2e-4, 0.00122, 9.5e10 },
	{ 196.0, 6.5e-4, 0.00335, 1.82e10 },
	{ 130.8, 7.5e-4, 0.00442, 6.54e9 }
    }
};

static const StringParameters celloStrings[3][4] = {
    {
	{ 220.0, 7.5e-4, 0.0017, 2.5e10 },
	{ 146.8, 8.8e-4, 0.0025, 2.5e10 },
	{ 98.0, 0.0012, 0.0062, 8.6e9 },
	{ 65.41, 0.0014, 0.0212, 2.24e10 }
    },
    {
	{ 220.0, 7.7e-4, 0.0016, 2.12e10 },
	{ 146.8, 0.001, 0.0025, 2.13e10 },
	{ 98.0, 0.0011, 0.0062, 1.62e10 },
	{ 65.41, 0.0015, 0.0175, 1.34e10 }
    },
    {
	{ 220.0, 7.7e-4, 0.0015, 2.25e10 },
	{ 146.8, 9.1e-4, 0.0036, 2.5e10 },
	{ 98.0, 0.0011, 0.0079, 2.22e10 },
	{ 65.41, 0.0014, 0.0192, 2.38e10 }
    }
};


// static tables shared between all instances
int BowedString::tablesize = 0;
double *BowedString::fric_table = NULL;
double *BowedString::dfric_table = NULL;
double *BowedString::intercept = NULL;
InterceptEntry *BowedString::interceptSorted = NULL;
double BowedString::fricmax = 0.0;
double BowedString::interceptmin = 0.0;

BowedString::BowedString(string name, double f0, double rho, double rad, double E, double T60_0, double T60_1000, double L)
    : Component1D(name)
{
    //act_min = -1e-3;
    //act_max = -2e-3;
    init(f0, rho, rad, E, T60_0, T60_1000, L);
}

BowedString::BowedString(string name, string type, int instrumentIndex, int stringIndex)
    : Component1D(name)
{
    double L, T60_0 = 10.0, T60_1000 = 5.0;
    const StringParameters *strp;

    if ((stringIndex < 0) || (stringIndex >= 4)) {
	logMessage(5, "Invalid string index %d", stringIndex);
	exit(1);
    }

    if (type == "violin") {
	if ((instrumentIndex < 0) || (instrumentIndex >= 5)) {
	    logMessage(5, "Violin index must be in range 0-4");
	    exit(1);
	}
	L = 0.32;
	//act_min = -1e-3;
	//act_max = -2e-3;
	strp = &violinStrings[instrumentIndex][stringIndex];
    }
    else if (type == "viola") {
	if ((instrumentIndex < 0) || (instrumentIndex >= 2)) {
	    logMessage(5, "Viola index must be in range 0-1");
	    exit(1);
	}
	L = 0.38;
	//act_min = -4e-3;
	//act_max = -6e-3;
	strp = &violaStrings[instrumentIndex][stringIndex];
    }
    else if (type == "cello") {
	if ((instrumentIndex < 0) || (instrumentIndex >= 3)) {
	    logMessage(5, "Cello index must be in range 0-2");
	    exit(1);
	}
	L = 0.69;
	//act_min = -6e-3;
	//act_max = -9e-3;
	strp = &celloStrings[instrumentIndex][stringIndex];
    }
    else {
	logMessage(5, "Unrecognised string instrument type '%s'", type.c_str());
	exit(1);
    }

    init(strp->f0, strp->rho, strp->diameter * 0.5, strp->E, T60_0, T60_1000, L);
}

// comparison function for sorting the intercepts using qsort
static int compareIntercept(const void *i1, const void *i2)
{
    InterceptEntry *ie1 = (InterceptEntry *)i1;
    InterceptEntry *ie2 = (InterceptEntry *)i2;
    if (ie1->value < ie2->value) return -1;
    if (ie1->value == ie2->value) return 0;
    return 1;
}

void BowedString::init(double f0, double rho, double rad, double E, double T60_0, double T60_1000, double L)
{
    logMessage(1, "Entering BowedString initialiser: %f, %f, %f, %f, %f, %f, %f", f0, rho,
	       rad, E, T60_0, T60_1000, L);

    B = NULL;
    C = NULL;

    numFingers = 0;

    // compute scalar parameters
    double a, c, Kappa, loss_coeff1, loss_coeff2, hmin;
    int N;
    int i, j;

    this->rho = rho;
    this->f0 = f0;
    this->L = L;
    this->E = E;
    this->rad = rad;

    k = 1.0 / GlobalSettings::getInstance()->getSampleRate();

    a = M_PI * rad * rad;
    I0 = M_PI * rad * rad * rad * rad / 4.0;
    c = 2.0 * L * f0;
    T = (c * c) * rho;
    Kappa = sqrt(E * I0 / rho);

    //loss_coeff1 = (-((c/L)*(c/L)) + sqrt(((c/L)*(c/L)*(c/L)*(c/L)))) / (2.0*Kappa*Kappa);
    loss_coeff1 = 0.0; // this is always going to be zero
    loss_coeff2 = (-((c*c)/(L*L)) + sqrt((pow(c/L, 4.0)) + 4.0*Kappa*Kappa*1000.0*1000.0)) / (2.0*Kappa*Kappa);

    lambda1 = 12.0 * log(10.0) / (loss_coeff2 - loss_coeff1) * ((loss_coeff2 / T60_0) - (loss_coeff1 / T60_1000));
    lambda2 = 12.0 * log(10.0) / (loss_coeff2 - loss_coeff1) * ((-1.0/T60_0) + (1.0/T60_1000));

    hmin = sqrt((c*c*k*k + 2.0*lambda2*k + sqrt((c*c*k*k + 2.0*lambda2*k)*(c*c*k*k + 2.0*lambda2*k) + 16.0*k*k*Kappa*Kappa)) / 2.0);
    N = (int)floor(L / hmin);
    h = L / (double)N;

    logMessage(1, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %d %.15f", a, I0, c, T, Kappa, loss_coeff1, loss_coeff2, lambda1, lambda2, hmin, N, h);

    // allocate state, including the extra polarisation
    allocateState(N-1);
    w = new double[ss];
    w1 = new double[ss];
    w2 = new double[ss];
    memset(w, 0, ss * sizeof(double));
    memset(w1, 0, ss * sizeof(double));
    memset(w2, 0, ss * sizeof(double));


    // generate update matrices B & C
    double Avals = 2.0 * k * k / (rho * (2.0 + lambda1 * k));
    double D = Avals * rho / (k*k);
    logMessage(1, "%.15f %.15f", Avals, D);

    // Dxm and Dxx need saved for energy calculation
    // Dxp and Dxxxx don't need saved

    Dxm = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Dxm, N, (N-1), 2*(N-1));
    for (i = 0; i < N; i++) {
	if (i < (N-1)) CSRSetValue(Dxm, i, i, (1.0 / h));
	if (i > 0) CSRSetValue(Dxm, i, (i-1), -(1.0 / h));
    }
    CSRmatrix *Dxp_temp = CSR_transpose(Dxm);
    CSR_scalar_mult(Dxp_temp, -1.0);
    Dxx = CSR_matrix_multiply(Dxp_temp, Dxm);
    CSRmatrix * Dxxxx_temp = CSR_matrix_multiply(Dxx, Dxx);

    CSR_scalar_mult(Dxxxx_temp, Kappa*Kappa*k*k);
    CSRmatrix *tmp1 = CSR_create_eye(N-1);
    CSR_scalar_mult(tmp1, 2.0);
    CSRmatrix *tmp2 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp2, (T*k*k/rho) + (lambda2*k));
    CSRmatrix *tmp3 = CSR_matrix_add(tmp1, tmp2);
    B = CSR_matrix_sub(tmp3, Dxxxx_temp);
    CSR_scalar_mult(B, D);

    CSR_free(Dxxxx_temp);
    CSR_free(Dxp_temp);
    CSR_free(tmp1);
    CSR_free(tmp2);
    CSR_free(tmp3);

    tmp1 = CSR_create_eye(N-1);
    CSR_scalar_mult(tmp1, (lambda1*k/2.0) - 1.0);
    tmp2 = CSR_duplicate(Dxx);
    CSR_scalar_mult(tmp2, lambda2*k);
    C = CSR_matrix_sub(tmp1, tmp2);
    CSR_scalar_mult(C, D);

    CSR_free(tmp1);
    CSR_free(tmp2);

    inv2k = 1.0 / (2.0*k);

    A = Avals;

    // backboard parameters for Newton
    /*KN = 1e10;
    alphaN = 1.5;
    betaN = 20.0;*/

    // finger parameters for Newton
    /*KF = 1e4;
    alphaF = 2.2;
    betaF = 200.0;
    MF = 0.02;*/

    // bow parameters for Newton
    KB = 1e5;
    alphaB = 2.0;
    betaB = 20.0;
    lambdaB = 10.0;
    MB = 0.1;

    epsr = 1e-13;
    tol = 2.2204460492503130808e-16;
    //tol = 1e-13;

    // make friction LUT
    // this is static and shared between all strings
    if (!fric_table) {
	double tol_F = 1e-6;
	tablesize = (int)(1.0 / tol_F);
    
	double *Vrel_vals = new double[tablesize];
	for (i = 0; i < tablesize; i++) {
	    Vrel_vals[i] = tol_F * ((double)(i+1));
	}
	fric_table = new double[tablesize+1];
	dfric_table = new double[tablesize+1];
	intercept = new double[tablesize+1];

	// Friction() function
	for (i = 0; i < tablesize; i++) {
	    double expvel10 = exp(-fabs(Vrel_vals[i]) * 10.0);
	    double expvel100 = pow(expvel10, 10.0);
	    double sgv = (Vrel_vals[i] >= 0.0) ? 1.0 : -1.0;
	    fric_table[i+1] = sgv * (0.4*expvel100 + 0.45*expvel10 + 0.35);
	    dfric_table[i+1] = -(40.0*expvel100 + 4.5*expvel10);
	}
	fric_table[0] = 1.2;
	dfric_table[0] = -44.5;

	// compute intercept
	// watch out! Vrel_vals has 0 added to start in Matlab code later on
	intercept[0] = fric_table[0];
	for (i = 1; i < (tablesize+1); i++) {
	    intercept[i] = -dfric_table[i] * Vrel_vals[i-1] + fric_table[i];
	}

	// compute fricmax and interceptmin
	fricmax = -1000000000.0;
	interceptmin = 1000000000.0;
	for (i = 0; i < (tablesize+1); i++) {
	    if (fric_table[i] > fricmax) fricmax = fric_table[i];
	    if (intercept[i] < interceptmin) interceptmin = intercept[i];
	}

	delete[] Vrel_vals;

	// sort intercept table!
	interceptSorted = new InterceptEntry[tablesize+1];
	for (i = 0; i < (tablesize+1); i++) {
	    interceptSorted[i].value = intercept[i];
	    interceptSorted[i].index = i;
	}
	qsort(interceptSorted, tablesize+1, sizeof(InterceptEntry), compareIntercept);
    }

    // initialise energy check
    energy = NULL;
    if (GlobalSettings::getInstance()->getEnergyOn()) {
	int Nf = GlobalSettings::getInstance()->getNumTimesteps();
	energy = new double[Nf];
	Hws = new double[Nf];
	Qws = new double[Nf];
	Hu = new double[Nf];
	Qus = new double[Nf];
	Ew = new double[Nf];
	Eu = new double[Nf];

	Hwc = new double[Nf];
	Hw = new double[Nf];
	Pw = new double[Nf];
	Qwc = new double[Nf];
	powsum_w = new double[Nf];
	Pu = new double[Nf];
	Quf = new double[Nf];
	powsum_u = new double[Nf];

	etmp1 = new double[ss+1];
	etmp2 = new double[ss+1];

	Hus = new double[Nf];
	HuB = new double[Nf];
	QuB = new double[Nf];

	Hws[0] = 0.0;
	Hu[0] = 0.0;
	Qws[0] = 0.0;
	Qus[0] = 0.0;
	Ew[0] = 0.0;
	Eu[0] = 0.0;

	Hwc[0] = 0.0;
	Hw[0] = 0.0;
	Pw[0] = 0.0;
	Qwc[0] = 0.0;
	powsum_w[0] = 0.0;
	Pu[0] = 0.0;
	Quf[0] = 0.0;
	powsum_u[0] = 0.0;

	Hus[0] = 0.0;
	HuB[0] = 0.0;
	QuB[0] = 0.0;

	energy[0] = 0.0;

	compw = 0.0;
	compu = 0.0;
    }
}

BowedString::~BowedString()
{
    int i;
    if (B) CSR_free(B);
    if (C) CSR_free(C);

    CSR_free(Dxx);
    CSR_free(Dxm);

    delete[] w;
    delete[] w1;
    delete[] w2;

    for (i = 0; i < bows.size(); i++) {
	delete bows[i].f_ext_w;
	delete bows[i].pos;
	delete bows[i].f_ext_u;
	if (bows[i].vibrato) {
	    delete[] bows[i].vibrato;
	}

	/*char wavname[1000];
	sprintf(wavname, "Vrel_%s_%d.wav", name.c_str(), (i+1));
	string filename = wavname;
	int j;
	double max = bows[i].Vrel_out->getMaxValue();
	WavWriter vrelwav(filename);
	vrelwav.writeMonoWavFile(bows[i].Vrel_out, max);

	delete bows[i].Vrel_out;*/
    }

    if (fric_table) {
	delete[] fric_table;
	delete[] dfric_table;
	delete[] intercept;

	delete[] interceptSorted;

	fric_table = NULL;
    }

    if (energy) {
	delete[] energy;
	delete[] Hws;
	delete[] Qws;
	delete[] Hu;
	delete[] Qus;
	delete[] Ew;
	delete[] Eu;
	delete[] Hwc;
	delete[] Hw;
	delete[] Pw;
	delete[] Qwc;
	delete[] powsum_w;
	delete[] Pu;
	delete[] Quf;
	delete[] powsum_u;
	delete[] etmp1;
	delete[] etmp2;
	delete[] Hus;
	delete[] HuB;
	delete[] QuB;
    }
}

void BowedString::runTimestep(int n)
{
    int i, j, ii;

    if (n < 2) return;

    // Basic string update
    CSR_matrix_vector_mult(B, w1, w);
    CSR_matrix_vector(C, w2, w, FALSE, OP_ADD);

    CSR_matrix_vector_mult(B, u1, u);
    CSR_matrix_vector(C, u2, u, FALSE, OP_ADD);

    int nobj = bows.size();

    // Bow update
    for (i = 0; i < nobj; i++) {
	Bow &bow = bows[i];

	// update horizontal bow position
	bow.yu = bow.By*bow.yu1 + bow.Cy*bow.yu2;

	// Compute J if bow position changed
	double newpos = bow.pos->getValue();
	if (bow.vibrato) newpos += bow.vibrato[n];
	if (newpos == bow.lastpos) {
	    // not changed, use from last timestep
	    bow.pos->next();
	    /* No need to do this - old values will still be in there anyway
	      bow.Ji = bow.Ji1;
	      bow.J[0] = bow.J1[0];
	      bow.J[1] = bow.J1[1];
	      bow.J[2] = bow.J1[2];
	      bow.J[3] = bow.J1[3]; */
	}
	else {
	    // changed, compute new one
	    bow.lastpos = newpos;
	    bow.Ji = getBowInterpInfo(bow, bow.J, n); // will call pos->next()
	}
    }

    if (nobj == 1) {
	// Scalar contact Newton
    	Bow &bow = bows[0];

	// Contact Newton
	double Kdelta2alm, P2prime;
	if (bow.delta2 < 0.0) {
	    Kdelta2alm = 0.0;
	    P2prime = 0.0;
	}
	else {
	    Kdelta2alm = bow.Kw * pow(bow.delta2, bow.alm);
	    //Kdelta2alm = bow.Kw * bow.delta2;
	    P2prime = Kdelta2alm * bow.delta2;
	}
	double P2sec = Kdelta2alm * bow.alpha;

	double md1 = max(bow.delta1, 0.0);
	double Psi = bow.Kb * pow(max(bow.delta1, 0.0), bow.alpha);
	//double Psi = bow.Kb * md1 * md1;

	// bw = 2*(y1 - y2) + k^2*Minv*(f_ext(n,:))' + hvecmat*(Jw'*updW - Jw2'*w2);
	double bw = 2.0 * (bow.yw1 - bow.yw2) + k*k*bow.f_ext_w->getValue()/bow.M;
	for (j = 0; j < 4; j++) {
	    // Jw is -J, Jw2 is -J2, so signs here are reversed
	    bw -= bow.J[j]  * w[bow.Ji+j-1];
	    bw += bow.J2[j] * w2[bow.Ji2+j-1];
	}

	// MC2 = AmatB*hvecmat*Jw'*Javw + k^2*Minv;
	// MC2 = 0.5*A*h*Jw'*Jw + 0.5*A*h*Jw'*Jw2 + k*k/MB
	// (Jw = -J, but the minus signs cancel when multiplying two J terms together)
	double MC2 = k*k/bow.M;
	double temp = 0.0;
	// add Jw term
	for (j = 0; j < 4; j++) {
	    temp += bow.J[j] * bow.J[j];
	}
	temp += interpolantProduct(bow.Ji, bow.Ji2, bow.J, bow.J2);
	bow.JJpJJ2 = temp;
	// our J doesn't have the divide by h factored into it. So that cancels out the
	// multiply by h that would be here, and requires a division by h instead
	MC2 += 0.5 * A * temp / h;

	double MC1 = 1.0 + inv2k * MC2 * Psi;

	double Fc = 0.0;
	int it = 0;
	int resetcount = 0;
	double diff = 1.0;
	double r = -bw/MC1; //-MC1 / bw;

	while (diff > tol) {
	    double maxr, rdelta2pos, Pprime, R, FuncC, Rprime, sgr;

	    if (fabs(r) > epsr) {
		// bigr
		maxr = fabs(r);
		rdelta2pos = max(r + bow.delta2, 0.0);
		Pprime = bow.Kw * pow(rdelta2pos, bow.alpha);
		//Pprime = bow.Kw * rdelta2pos * rdelta2pos;
		bow.P = Pprime * rdelta2pos / (bow.alpha + 1.0);
		sgr = r >= 0.0 ? 1.0 : -1.0;
		R = sgr * (bow.P - bow.P2) / maxr;
		FuncC = MC1 * r + MC2 * R + bw;
		Rprime = (r * Pprime - bow.P + bow.P2) / (maxr*maxr);
	    }
	    else {
		// smallr
		maxr = epsr;
		rdelta2pos = max(r + bow.delta2, 0.0);
		Pprime = bow.Kw * pow(rdelta2pos, bow.alpha);
		//Pprime = bow.Kw * rdelta2pos * rdelta2pos;
		bow.P = Pprime * rdelta2pos / (bow.alpha + 1.0);
		R = P2prime;
		FuncC = MC1*r + MC2*R + bw;
		Rprime = P2sec;
	    }
	    temp = Rprime;
	    double JacC = MC1 + MC2*temp;

	    double rnew = r - FuncC / JacC; //r - JacC / FuncC;
	    diff = fabs(r - rnew);
	    r = rnew;
	    it = it + 1;
	    if (it > 50) {
		r = 0.0;
		resetcount++;
		if (resetcount > 1) break;
		it = 0;
	    }
	}

	bow.delta = r + bow.delta2;
	md1 = max(bow.delta, 0.0);
	bow.P = bow.Ka1 * pow(max(bow.delta, 0.0), bow.alp);
	//bow.P = bow.Ka1 * md1 * md1 * md1;
	if (fabs(r) > epsr) {
	    // bigr
	    // original is sign(r)*(P-P2)/max(abs(r),epsr). but since we already know
	    // that abs(r) > epsr, we can replace with (P-P2)/r
	    Fc = (bow.P - bow.P2) / r + r * Psi * inv2k;
	}
	else {
	    // smallr
	    Fc = P2prime + r * Psi * inv2k;
	}
	bow.Fc = Fc;
	bow.Psi = Psi;
    }
    else {
	// Vector contact Newton!
	double Kdelta2alm[MAX_OBJS], P2prime[MAX_OBJS];
	double P2sec[MAX_OBJS], md1[MAX_OBJS], Psi[MAX_OBJS];
	double bw[MAX_OBJS];
	double MC1[MAX_OBJS*MAX_OBJS], MC2[MAX_OBJS*MAX_OBJS];
	double Fc[MAX_OBJS];
	double diff[MAX_OBJS];
	double r[MAX_OBJS], maxr[MAX_OBJS], rdelta2pos[MAX_OBJS];
	double Pprime[MAX_OBJS], R[MAX_OBJS], FuncC[MAX_OBJS];
	double Rprime[MAX_OBJS], rnew[MAX_OBJS];
	double JacC[MAX_OBJS*MAX_OBJS];

	double L[MAX_OBJS*MAX_OBJS], U[MAX_OBJS*MAX_OBJS];
	double cy[MAX_OBJS];

	int it = 0;
	int resetcount = 0;

	for (i = 0; i < nobj; i++) {
	    Bow &bow = bows[i];
	    if (bow.delta2 < 0.0) {
		Kdelta2alm[i] = 0.0;
		P2prime[i] = 0.0;
	    }
	    else {
		Kdelta2alm[i] = bow.Kw * pow(bow.delta2, bow.alm);
		P2prime[i] = Kdelta2alm[i] * bow.delta2;
	    }
	    P2sec[i] = Kdelta2alm[i] * bow.alpha;

	    Psi[i] = bow.Kb * pow(max(bow.delta1, 0.0), bow.alpha);

	    bw[i] = 2.0 * (bow.yw1 - bow.yw2) + k*k*bow.f_ext_w->getValue()/bow.M;
	    for (j = 0; j < 4; j++) {
		// Jw is -J, Jw2 is -J2, so signs here are reversed
		bw[i] -= bow.J[j]  * w[bow.Ji+j-1];
		bw[i] += bow.J2[j] * w2[bow.Ji2+j-1];
	    }

	    // need to compute JJpJJ2 for the friction code later
	    double temp = 0.0;
	    // add Jw term
	    for (j = 0; j < 4; j++) {
		temp += bow.J[j] * bow.J[j];
	    }
	    temp += interpolantProduct(bow.Ji, bow.Ji2, bow.J, bow.J2);
	    bow.JJpJJ2 = temp;

	    Fc[i] = 0.0;
	    diff[i] = 1.0;
	}

	// FIXME: possibly only use this when it's really needed
	// (i.e. when two objects overlap)

	// Now calculate MC2!
	// MC2 = AmatB*hvecmat*Jw'*JavW + k^2*Minv;
	// MC2 = 0.5*A*h*Jw'*Jw + 0.5*A*h*Jw'*Jw2 + k*k/M
	for (i = 0; i < nobj; i++) {
	    for (j = 0; j < nobj; j++) {
		double temp = 0.0;
		temp += interpolantProduct(bows[i].Ji, bows[j].Ji, bows[i].J, bows[j].J);
		temp += interpolantProduct(bows[i].Ji, bows[j].Ji2, bows[i].J, bows[j].J2);

		// FIXME: is this the right way round??
		MC2[(j*nobj)+i] = 0.5 * A * temp / h;
	    }
	    
	    // add to diagonal
	    MC2[(i*nobj)+i] += (k*k) / bows[i].M;
	}

	// Calculate MC1
	// MC1 = speye(Nobj) + inv2k*MC2*Psidiag;
	for (i = 0; i < nobj; i++) {
	    for (j = 0; j < nobj; j++) {
		MC1[(j*nobj)+i] = inv2k * MC2[(j*nobj)+i] * Psi[j];
	    }
	    MC1[(i*nobj)+i] += 1.0;
	}

	// Calculate initial r
	// r = -MC1\bw
	croutDecomposition(MC1, L, U, nobj);
	croutSolve(L, U, bw, r, cy, nobj);
	for (i = 0; i < nobj; i++) {
	    r[i] = -r[i];
	}

	// solver loop
	bool done = false;
	while (!done) {
	    double sgr, temp;

      	    // compute R
	    for (i = 0; i < nobj; i++) {
		if (fabs(r[i] > epsr)) {
		    maxr[i] = fabs(r[i]);
		    rdelta2pos[i] = max(r[i] + bows[i].delta2, 0.0);
		    Pprime[i] = bows[i].Kw * pow(rdelta2pos[i], bows[i].alpha);
		    bows[i].P = Pprime[i] * rdelta2pos[i] / (bows[i].alpha + 1.0);
		    sgr = r[i] >= 0.0 ? 1.0 : -1.0;
		    R[i] = sgr * (bows[i].P - bows[i].P2) / maxr[i];
		    Rprime[i] = (r[i] * Pprime[i] - bows[i].P + bows[i].P2) / (maxr[i]*maxr[i]);
		}
		else {
		    maxr[i] = epsr;
		    rdelta2pos[i] = max(r[i] + bows[i].delta2, 0.0);
		    Pprime[i] = bows[i].Kw * pow(rdelta2pos[i], bows[i].alpha);
		    bows[i].P = Pprime[i] * rdelta2pos[i] / (bows[i].alpha + 1.0);
		    R[i] = P2prime[i];
		    
		    Rprime[i] = P2sec[i];
		}
	    }

	    // compute FuncC
	    // FuncC = MC1*r + MC2*R + bw
	    for (i = 0; i < nobj; i++) {
		temp = bw[i];
		for (j = 0; j < nobj; j++) {
		    temp += MC1[(j*nobj)+i] * r[j];
		    temp += MC2[(j*nobj)+i] * R[j];
		}
		FuncC[i] = temp;
	    }

	    // compute JacC
	    // JacC = MC1 + MC2*temp
	    // temp is diagonal version of Rprime
	    // post-multiplication, so multiplies columns
	    for (i = 0; i < nobj; i++) {
		for (j = 0; j < nobj; j++) {
		    JacC[(j*nobj)+i] = MC1[(j*nobj)+i] + MC2[(j*nobj)+i] * Rprime[j];
		}
	    }
	    
	    // linear system solve
	    // rnew = r - JacC \ FuncC
	    croutDecomposition(JacC, L, U, nobj);
	    croutSolve(L, U, FuncC, rnew, cy, nobj);

	    // update r and diff
	    for (i = 0; i < nobj; i++) {
		rnew[i] = r[i] - rnew[i];
		diff[i] = fabs(r[i] - rnew[i]);
		r[i] = rnew[i];
	    }

	    // check if done yet
	    it = it + 1;
	    if (it > 50) {
		if (resetcount) break;
		resetcount++;
		it = 0;
		for (i = 0; i < nobj; i++) {
		    r[i] = 0.0;
		}
	    }

	    done = true;
	    for (i = 0; i < nobj; i++) {
		if (diff[i] > tol) {
		    done = false;
		    break;
		}
	    }
	}

	// update delta, P, Fc, Psi
	for (i = 0; i < nobj; i++) {
	    bows[i].delta = r[i] + bows[i].delta2;
	    bows[i].P = bows[i].Ka1 * pow(max(bows[i].delta, 0.0), bows[i].alp);
	    if (fabs(r[i]) > epsr) {
		Fc[i] = (bows[i].P - bows[i].P2) / r[i] + r[i] * Psi[i] * inv2k;
	    }
	    else {
		Fc[i] = P2prime[i] + r[i] * Psi[i] * inv2k;
	    }
	    bows[i].Fc = Fc[i];
	    bows[i].Psi = Psi[i];
	}
    }

    for (i = 0; i < nobj; i++) {
	Bow &bow = bows[i];

	// if we have multiple fingers, do them together in a vector solve
	if ((numFingers > 1) && (bow.isFinger)) continue;

	// Friction Newton
	double Ff = 0;
	double fric;

	if (bow.Fc != 0.0) {
	    double MF2;
	    MF2 = (0.5 * inv2k * A * bow.JJpJJ2 / h) + inv2k*bow.Dy;
	    // Ju2'*u2 - Ju'*u term
	    //bow.bu = bow.vB->getValue();
	    bow.bu = inv2k * ((bow.yu - bow.yu2) + bow.Dy*bow.f_ext_u->getValue());

	    // multiply by inv2k. no need to multiply by h, our J isn't divided by h
	    for (j = 0; j < 4; j++) {
		bow.bu += inv2k * bow.J2[j] * u2[bow.Ji2+j-1];
		bow.bu -= inv2k * bow.J[j]  * u [bow.Ji +j-1];
	    }

	    if (!bow.isFinger) {
		// do Friedlander. update Vrel and fric
		friedlander(fric, bow.Fc, MF2, bow);
	    }
	    else {
		// update bu with bow stuff if bow also on string!
		// need to deal with any bows that come close to overlapping this
		for (j = 0; j < nobj; j++) {
		    if (!bows[j].isFinger) {
			double temp = 0.0;
			temp += 0.5 * interpolantProduct(bows[j].Ji, bow.Ji,
							 bows[j].J, bow.J);
			temp += 0.5 * interpolantProduct(bows[j].Ji2, bow.Ji,
							 bows[j].J2, bow.J);
			bow.bu += (inv2k * A * temp * bows[j].Ff) / h;
		    }
		}

		coulombNewton(fric, bow.Fc, MF2, bow);
	    }

	    Ff = fric * max(bow.Fc, 0.0);
	}
	else {
	    bow.Vrel = -bow.bu;
	    fric = 0.0;
	}

	// Update u and w with values from this bow
	for (j = 0; j < 4; j++) {
	    // w update is a + in Matlab, but remember Jw* is negated compared to J*
	    w[bow.Ji+j-1]  -= 0.5 * A * bow.J[j]  * bow.Fc / h;
	    w[bow.Ji2+j-1] -= 0.5 * A * bow.J2[j] * bow.Fc / h;
	    u[bow.Ji+j-1]  -= 0.5 * A * bow.J[j]  * Ff / h;
	    u[bow.Ji2+j-1] -= 0.5 * A * bow.J2[j] * Ff / h;
	}

	bow.yw = (k*k*(bow.Fc + bow.f_ext_w->getValue()) / bow.M) + 2.0*bow.yw1 - bow.yw2;
	bow.yu = bow.yu + bow.Dy*(Ff + bow.f_ext_u->getValue());

	// keep this for the energy check
	bow.Ff = Ff;

	//bow.Vrel_out->getData()[n] = bow.Vrel;
    }

    if (numFingers > 1) {
	double fric[MAX_OBJS], Fcv[MAX_OBJS], MF2[MAX_OBJS*MAX_OBJS];
	double Vrel[MAX_OBJS], bu[MAX_OBJS];
	int fing[MAX_OBJS];

	// do vector friction solve for fingers

	// copy Fc and Vrel, compute bu
	ii = 0;
	for (i = 0; i < nobj; i++) {
	    if (bows[i].isFinger) {
		Bow &bow = bows[i];

		fing[ii] = i;

		Fcv[ii] = bows[i].Fc;
		Vrel[ii] = bows[i].Vrel;

		// get bu
		bow.bu = inv2k * ((bow.yu - bow.yu2) + bow.Dy*bow.f_ext_u->getValue());

		// multiply by inv2k. no need to multiply by h, our J isn't divided by h
		for (j = 0; j < 4; j++) {
		    bow.bu += inv2k * bow.J2[j] * u2[bow.Ji2+j-1];
		    bow.bu -= inv2k * bow.J[j]  * u [bow.Ji +j-1];
		}

		// need to deal with any bows that come close to overlapping this
		// finger
		for (j = 0; j < nobj; j++) {
		    if (!bows[j].isFinger) {
			double temp = 0.0;
			temp += 0.5 * interpolantProduct(bows[j].Ji, bow.Ji,
							 bows[j].J, bow.J);
			temp += 0.5 * interpolantProduct(bows[j].Ji2, bow.Ji,
							 bows[j].J2, bow.J);
			bow.bu += (inv2k * A * temp * bows[j].Ff) / h;
		    }
		}

		// store ready for vector solver
		bu[ii] = bow.bu;
		
		ii++;
	    }
	}

	// compute MF2 matrix
	// MF2 = inv2k*(AmatFB*hvecmat*Ju'*Javu) + inv2k*Dy
	for (i = 0; i < numFingers; i++) {
	    for (j = 0; j < numFingers; j++) {
		double temp = 0.0;
		temp += interpolantProduct(bows[fing[i]].Ji, bows[fing[j]].Ji, bows[fing[i]].J, bows[fing[j]].J);
		temp += interpolantProduct(bows[fing[i]].Ji, bows[fing[j]].Ji2, bows[fing[i]].J, bows[fing[j]].J2);
		MF2[(j*numFingers)+i] = inv2k * 0.5 * A * temp / h;
	    }
	    MF2[(i*numFingers)+i] += inv2k * bows[fing[i]].Dy;
	}

	// do the actual solve
	coulombNewton(fric, Fcv, MF2, Vrel, bu);

	// update Ff, w, u, yw, yu for each finger
	for (i = 0; i < numFingers; i++) {
	    Bow &bow = bows[fing[i]];

	    bow.Ff = fric[i] * max(bow.Fc, 0.0);

	    if (bow.Fc != 0.0) {
		bow.Vrel = Vrel[i];
	    }
	    else {
		bow.Vrel = -bow.bu;
	    }

	    // Update u and w with values from this bow
	    for (j = 0; j < 4; j++) {
		// w update is a + in Matlab, but remember Jw* is negated compared to J*
		w[bow.Ji+j-1]  -= 0.5 * A * bow.J[j]  * bow.Fc / h;
		w[bow.Ji2+j-1] -= 0.5 * A * bow.J2[j] * bow.Fc / h;
		u[bow.Ji+j-1]  -= 0.5 * A * bow.J[j]  * bow.Ff / h;
		u[bow.Ji2+j-1] -= 0.5 * A * bow.J2[j] * bow.Ff / h;
	    }
	    
	    bow.yw = (k*k*(bow.Fc + bow.f_ext_w->getValue()) / bow.M) + 2.0*bow.yw1 - bow.yw2;
	    bow.yu = bow.yu + bow.Dy*(bow.Ff + bow.f_ext_u->getValue());
	}
    }

    if (energy) {
	double powtemp, powsum_wtemp, powsum_utemp;

	// compute Hws
	Hws[n-1] = 0.0;
	for (i = 0; i < ss; i++) {
	    Hws[n-1] += 0.5 * h * rho * ((w[i] - w1[i])/k) * ((w[i] - w1[i])/k);
	}
	CSR_matrix_vector_mult(Dxm, w1, etmp1);
	CSR_matrix_vector_mult(Dxm, w,  etmp2);
	for (i = 0; i < (ss+1); i++) {
	    Hws[n-1] += 0.5 * T * h * etmp1[i] * etmp2[i];
	}
	CSR_matrix_vector_mult(Dxx, w1, etmp1);
	CSR_matrix_vector_mult(Dxx, w,  etmp2);
	for (i = 0; i < ss; i++) {
	    Hws[n-1] += 0.5 * E * I0 * h * etmp1[i] * etmp2[i];
	}
	CSR_matrix_vector_mult(Dxm, w, etmp1);
	CSR_matrix_vector(Dxm, w1, etmp1, FALSE, OP_SUB);
	for (i = 0; i < (ss+1); i++) {
	    Hws[n-1] -= 0.25/k * lambda2 * rho * h * etmp1[i] * etmp1[i];
	}
	
	// compute Qws
	Qws[n-1] = 0.0;
	for (i = 0; i < ss; i++) {
	    Qws[n-1] += lambda1 * h * rho * ((w[i] - w2[i]) * inv2k) * ((w[i] - w2[i]) * inv2k);
	}
	CSR_matrix_vector_mult(Dxm, w, etmp1);
	CSR_matrix_vector(Dxm, w2, etmp1, FALSE, OP_SUB);
	for (i = 0; i < (ss+1); i++) {
	    Qws[n-1] += lambda2 * rho * h * (etmp1[i]*inv2k) * (etmp1[i]*inv2k);
	}

	// Compute vertical bow energy
	Hwc[n-1] = 0.0;
	Pw[n-1] = 0.0;
	Qwc[n-1] = 0.0;
	for (i = 0; i < nobj; i++) {
	    Bow &bow = bows[i];
	    Hwc[n-1] += 0.5 * (bow.P + bow.P1) + (0.5/(k*k)) * bow.M * (((bow.yw - bow.yw1)) *
									((bow.yw - bow.yw1)));
	    Pw[n-1] += (inv2k * (bow.yw - bow.yw2)) * bow.f_ext_w->getValue();

	    // Pw -= (0.5*(hvecN.*(w + w2))'*inv2k*(Jw - Jw2))*Fc;
	    for (j = 0; j < 4; j++) {
		Pw[n-1] += 0.5 * inv2k * bow.Fc * bow.J[j] * (w[bow.Ji+j-1] + w2[bow.Ji+j-1]);
		Pw[n-1] -= 0.5 * inv2k * bow.Fc * bow.J2[j] * (w[bow.Ji2+j-1] + w2[bow.Ji2+j-1]);
	    }

	    Qwc[n-1] += (inv2k*(bow.delta-bow.delta2)) *
		(inv2k*(bow.delta-bow.delta2)) * bow.Psi;
	}
	Hw[n-1] = Hws[n-1] + Hwc[n-1];

	// compute Ew
	powtemp = k * (Pw[n-1] - Qws[n-1] - Qwc[n-1]) - compw;
	powsum_wtemp = powsum_w[n-2] + powtemp;
	compw = (powsum_wtemp - powsum_w[n-2]) - powtemp;
	powsum_w[n-1] = powsum_wtemp;
	Ew[n-1] = (Hw[n-1] - Hw[1] - (powsum_w[n-1] - powsum_w[1]));

	// compute Hu
	Hus[n-1] = 0.0;
	for (i = 0; i < ss; i++) {
	    Hus[n-1] += 0.5 * h * rho * ((u[i] - u1[i])/k) * ((u[i] - u1[i])/k);
	}
	CSR_matrix_vector_mult(Dxm, u1, etmp1);
	CSR_matrix_vector_mult(Dxm, u,  etmp2);
	for (i = 0; i < (ss+1); i++) {
	    Hus[n-1] += 0.5 * T * h * etmp1[i] * etmp2[i];
	}
	CSR_matrix_vector_mult(Dxx, u1, etmp1);
	CSR_matrix_vector_mult(Dxx, u,  etmp2);
	for (i = 0; i < ss; i++) {
	    Hus[n-1] += 0.5 * E * I0 * h * etmp1[i] * etmp2[i];
	}
	CSR_matrix_vector_mult(Dxm, u, etmp1);
	CSR_matrix_vector(Dxm, u1, etmp1, FALSE, OP_SUB);
	for (i = 0; i < (ss+1); i++) {
	    Hus[n-1] -= 0.25/k * lambda2 * rho * h * etmp1[i] * etmp1[i];
	}

	// compute Qus
	Qus[n-1] = 0.0;
	for (i = 0; i < ss; i++) {
	    Qus[n-1] += lambda1 * h * rho * ((u[i] - u2[i]) * inv2k) * ((u[i] - u2[i]) * inv2k);
	}
	CSR_matrix_vector_mult(Dxm, u, etmp1);
	CSR_matrix_vector(Dxm, u2, etmp1, FALSE, OP_SUB);
	for (i = 0; i < (ss+1); i++) {
	    Qus[n-1] += lambda2 * rho * h * (etmp1[i]*inv2k) * (etmp1[i]*inv2k);
	}

	// compute horizontal bow energy
	Pu[n-1] = 0.0;
	Quf[n-1] = 0.0;
	HuB[n-1] = 0.0;
	QuB[n-1] = 0.0;
	for (i = 0; i < nobj; i++) {
	    Bow &bow = bows[i];

	    HuB[n-1] += 0.5*(bow.M*(bow.yu - bow.yu1)/k)*((bow.yu - bow.yu1)/k) +
		0.25 * bow.Ku * (bow.yu*bow.yu + bow.yu1*bow.yu1);

	    for (j = 0; j < 4; j++) {
		Pu[n-1] += 0.5 * inv2k * bow.Ff * bow.J[j]  * (u[bow.Ji+j-1]  + u2[bow.Ji+j-1]);
		Pu[n-1] -= 0.5 * inv2k * bow.Ff * bow.J2[j] * (u[bow.Ji2+j-1] + u2[bow.Ji2+j-1]);
	    }
	    //Pu[n-1] -= (bow.vB->getValue() * bow.Ff);
	    Pu[n-1] += ((bow.yu - bow.yu2)*inv2k) * bow.f_ext_u->getValue();
	    Quf[n-1] += bow.Vrel * bow.Ff;
	    QuB[n-1] += bow.lambda * ((bow.yu - bow.yu2)*inv2k) *
		((bow.yu - bow.yu2)*inv2k);
	}

	Hu[n-1] = Hus[n-1] + HuB[n-1];

	// compute Eu
	powtemp = k * (Pu[n-1] - Qus[n-1] - Quf[n-1] - QuB[n-1]) - compu;
	powsum_utemp = powsum_u[n-2] + powtemp;
	compu = (powsum_utemp - powsum_u[n-2]) - powtemp;
	powsum_u[n-1] = powsum_utemp;
	Eu[n-1] = (Hu[n-1] - Hu[1] - (powsum_u[n-1] - powsum_u[1]));

	energy[n-1] = powsum_w[n-1] + powsum_u[n-1];
    }

    for (i = 0; i < nobj; i++) {
	bows[i].f_ext_w->next();
	bows[i].f_ext_u->next();
    }
}

void BowedString::swapBuffers(int n)
{
    if (n < 2) return;

    Component::swapBuffers(n);
    double *tmp = w2;
    w2 = w1;
    w1 = w;
    w = tmp;

    int i;
    for (i = 0; i < bows.size(); i++) {
	bows[i].yw2 = bows[i].yw1;
	bows[i].yw1 = bows[i].yw;

	bows[i].yu2 = bows[i].yu1;
	bows[i].yu1 = bows[i].yu;

	bows[i].Ji2 = bows[i].Ji1;
	bows[i].Ji1 = bows[i].Ji;

	bows[i].delta2 = bows[i].delta1;
	bows[i].delta1 = bows[i].delta;

	bows[i].J2[0] = bows[i].J1[0];
	bows[i].J2[1] = bows[i].J1[1];
	bows[i].J2[2] = bows[i].J1[2];
	bows[i].J2[3] = bows[i].J1[3];

	bows[i].J1[0] = bows[i].J[0];
	bows[i].J1[1] = bows[i].J[1];
	bows[i].J1[2] = bows[i].J[2];
	bows[i].J1[3] = bows[i].J[3];

	bows[i].P2 = bows[i].P1;
	bows[i].P1 = bows[i].P;
    }
}

void BowedString::logMatrices()
{
    if (B) saveMatrix(B, "B");
    if (C) saveMatrix(C, "C");
}

// convenience method for multiplying two sets of interpolation info together
double BowedString::interpolantProduct(int idx1, int idx2, double *J1, double *J2)
{
    double temp = 0.0;

    switch (idx1 - idx2) {
    case -3:
	// idx2 was 3 greater
	temp += J1[3] * J2[0];
	break;
    case -2:
	// idx2 was 2 greater
	temp += J1[2] * J2[0];
	temp += J1[3] * J2[1];
	break;
    case -1:
	// idx2 was 1 greater
	temp += J1[1] * J2[0];
	temp += J1[2] * J2[1];
	temp += J1[3] * J2[2];
	break;
    case 0:
	// exact match up
	temp += J1[0] * J2[0];
	temp += J1[1] * J2[1];
	temp += J1[2] * J2[2];
	temp += J1[3] * J2[3];
	break;
    case 1:
	// idx1 was 1 greater
	temp += J1[0] * J2[1];
	temp += J1[1] * J2[2];
	temp += J1[2] * J2[3];
	break;
    case 2:
	// idx1 was 2 greater
	temp += J1[0] * J2[2];
	temp += J1[1] * J2[3];
	break;
    case 3:
	// idx1 was 3 greater
	temp += J1[0] * J2[3];
	break;
    }    
    return temp;
}


// returns start index (pos_int)
// the co-effs don't include the division by h
int BowedString::getBowInterpInfo(Bow &bow, double coeffs[4], int n)
{
    double N = (double)(ss + 1);
    double pos = bow.pos->getValue();
    bow.pos->next();

    if (bow.vibrato) pos += bow.vibrato[n];

    int pos_int = (int)floor(pos * N);
    double pos_frac = (pos * N) - (double)pos_int;

    coeffs[0] = -pos_frac * (pos_frac-1.0) * (pos_frac - 2.0) / 6.0;
    coeffs[1] = (pos_frac - 1.0) * (pos_frac + 1.0) * (pos_frac - 2.0) * 0.5;
    coeffs[2] = -pos_frac * (pos_frac + 1.0) * (pos_frac - 2.0) * 0.5;
    coeffs[3] = pos_frac * (pos_frac + 1.0) * (pos_frac - 1.0) / 6.0;

    return pos_int;
}


void BowedString::addBow(double w0, double vw0, double u0, double vu0,
			 vector<double> *times, vector<double> *positions,
			 vector<double> *forces_w, vector<double> *forces_u,
			 vector<double> *vibrato, bool isFinger)
{
    logMessage(1, "Adding bow, %.15f %.15f %.15f %.15f", w0, vw0, u0, vu0);

    Bow bow;
    int i, j;

    bow.yw = 0.0;
    bow.yw1 = w0 + (vw0 * k);
    bow.yw2 = w0;

    bow.yu = 0.0;
    bow.yu1 = u0 + (vu0 * k);
    bow.yu2 = u0;

    bow.f_ext_w = new BreakpointFunction(times->data(), forces_w->data(), forces_w->size(), k);
    bow.pos = new BreakpointFunction(times->data(), positions->data(), positions->size(), k);
    bow.f_ext_u = new BreakpointFunction(times->data(), forces_u->data(), forces_u->size(), k);

    bow.vibrato = NULL;

    bow.isFinger = isFinger;
    if (vibrato != NULL) {
	// create the vibrato adjustment for each timestep
	int Nf = GlobalSettings::getInstance()->getNumTimesteps();
	double Fs = GlobalSettings::getInstance()->getSampleRate();
	bow.vibrato = new double[Nf];
	memset(bow.vibrato, 0, Nf * sizeof(double));

	for (i = 0; i < vibrato->size(); i += 5) {
	    // convert timing values to timesteps
	    int t1 = (int)(vibrato->at(i+0) * Fs);
	    int t2 = (int)(vibrato->at(i+1) * Fs);
	    int ramp = (int)(vibrato->at(i+2) * Fs);
	    double amplitude = vibrato->at(i+3);
	    double freq = vibrato->at(i+4);
	    int tt = t1 + ramp;
	    
	    for (j = 0; j < ramp; j++) {
		if ((t1+j) < Nf) {
		    bow.vibrato[t1+j] = (0.7*amplitude/
					 ((double)ramp)*((double)(t1+j-tt)) +
					 amplitude) * sin(2.0 * M_PI * freq *
							  ((double)j) / Fs);
		}
	    }
	    for (j = ramp; j < (t2-t1); j++) {
		if ((t1+j) < Nf) {
		    bow.vibrato[t1+j] = amplitude * sin(2.0 * M_PI * freq *
							((double)j) / Fs);
		}
	    }
	}
    }

    if (!isFinger) {
	bow.Kw = KB;
	bow.Ku = 0.0;
	bow.alpha = alphaB;
	bow.beta = betaB;
	bow.lambda = lambdaB;
	bow.M = MB;
    }
    else {
	bow.Kw = KwF;
	bow.Ku = KuF;
	bow.alpha = alphaF;
	bow.beta = betaF;
	bow.lambda = lambdaF;
	bow.M = MF;

	numFingers++;
    }
    bow.alp = bow.alpha + 1.0;
    bow.alm = bow.alpha - 1.0;
    bow.Ka1 = bow.Kw / bow.alp;
    bow.Ka = bow.Kw * bow.alpha;
    bow.Kb = bow.Kw * bow.beta;

    // calculate first two Js
    bow.Ji2 = getBowInterpInfo(bow, bow.J2, 0);
    bow.Ji1 = getBowInterpInfo(bow, bow.J1, 1);

    // skip first two timesteps on all the other breakpoint functions
    bow.f_ext_w->next();
    bow.f_ext_w->next();
    bow.f_ext_u->next();
    bow.f_ext_u->next();

    // in original code, delta1 = -hvecmat*Jw1'*w1 - y1, but w1 is always zero at this point
    bow.delta1 = -bow.yw1;
    bow.delta2 = -bow.yw2;
    bow.delta = 0.0;

    double Kdelta1alpha = pow((bow.Kw * (max(bow.delta1, 0.0))), bow.alpha);
    bow.P1 = Kdelta1alpha * max(bow.delta1, 0.0) / bow.alp;
    bow.P2 = pow((bow.Ka1 * (max(bow.delta2, 0.0))), bow.alp);

    bow.lastpos = -1.0;

    bow.bu = 0.0;
    bow.Vrel = 0.0;

    // create By, Cy, Dy for updates
    bow.By = 2.0;
    bow.Cy = -1.0 + 0.5 * (k*bow.lambda - k*k*bow.Ku) / bow.M;
    bow.Dy = 1.0 / ((2.0*bow.M + k*bow.lambda + k*k*bow.Ku) * (0.5 / bow.M));
    bow.By = bow.By * bow.Dy;
    bow.Cy = bow.Cy * bow.Dy;
    bow.Dy = (bow.Dy * k * k) / bow.M;

    //bow.Vrel_out = new Output(this, 0.5, 0.0);

    bows.push_back(bow);

    if (bows.size() >= MAX_OBJS) {
	logMessage(5, "Error: too many bows and fingers on a single string (maximum is %d)", MAX_OBJS);
	exit(1);
    }

    Input::setFirstInputTimestep(2);

    if (energy) {
	// initial energy calculation
	Hwc[0] += 0.5 * (bow.P1 + bow.P2) + (0.5/(k*k)) * ((bow.M * (bow.yw1 - bow.yw2)) *
							   (bow.M * (bow.yw1 - bow.yw2)));
	Hw[0] = Hwc[0];

	HuB[0] += 0.5 * bow.M * ((bow.yu1 - bow.yu2)/k) * ((bow.yu1 - bow.yu2) / k);
	Hu[0] = HuB[0];
    }
}

void BowedString::addFinger(double w0, double vw0, double u0, double vu0,
			    vector<double> *times, vector<double> *positions,
			    vector<double> *forces_w, vector<double> *forces_u,
			    vector<double> *vibrato)
{
    logMessage(1, "Adding finger, %.15f %.15f %.15f %.15f", w0, vw0, u0, vu0);
    addBow(w0, vw0, u0, vu0, times, positions, forces_w, forces_u, vibrato, true);
}


// Matlab F is Fc
// Matlab b is bow.bu
// Matlab MF is MF2
// Other Matlab values are instance variables (friction table etc.)
// bow.Vrel is both input and output. fric is just output. other parameters are just inputs
void BowedString::friedlander(double &fric, double Fc, double MF2, Bow &bow)
{
    int i;
    double Vrel_old = bow.Vrel;
    bow.Vrel = 0.0;
    fric = 0.0;

    if (Fc > 0.0) {
	double p = -bow.bu / (MF2 * Fc);
	double ppos = fabs(p);
	double sgn = p >= 0.0 ? 1.0 : -1.0;
	if (ppos > fricmax) {
	    // no ambiguity, slipping
	    Newton_friction(bow.Vrel, fric, Fc, bow.bu, MF2, sgn);
	}
	else if (ppos <= interceptmin) {
	    // no ambiguity, sticking
	    bow.Vrel = 0.0;
	    fric = p;
	}
	else {
	    // ambiguity
	    // find tangent that intersects vertical axis at intercept

	    // binary search the sorted intercept table
	    int blockstart = 0;
	    int blockend = tablesize;
	    int partition = tablesize / 2;
	    int blocksize = tablesize + 1;
	    while (blocksize > 2) {
		if (ppos >= interceptSorted[partition].value) {
		    // in top half
		    blockstart = partition;
		}
		else {
		    // in bottom half
		    blockend = partition - 1;
		}
		blocksize = (blockend - blockstart) + 1;
		partition = (blockstart + blockend) / 2;
	    }

	    // blocksize must now be 2
	    int xint = -1;
	    if (ppos >= interceptSorted[partition].value) {
		xint = interceptSorted[blockstart+1].index;
	    }
	    else {
		xint = interceptSorted[blockstart].index;
	    }

	    /*double pval = 1000000000.0;
	    for (i = 0; i < (tablesize+1); i++) {
		double dist = ppos - intercept[i];
		if ((dist < pval) && (dist > 0.0)) {
		    pval = dist;
		    xint = i;
		}
		}*/

	    xint--;
	    double xfrac = (ppos - intercept[xint]) / (intercept[xint+1] - intercept[xint]);
	    double slope = LagrangeInterp(xint, xfrac);
	    double m = p / bow.bu;

	    if (m < slope) {
		// no ambiguity, sticking
		bow.Vrel = 0.0;
		fric = p;
	    }
	    else if ((m >= slope) && (Vrel_old == 0.0)) {
		// ambiguity
		// previous state was sticking
		bow.Vrel = 0.0;
		fric = p;
	    }
	    else {
		// ambiguity, previous state was slipping
		Newton_friction(bow.Vrel, fric, Fc, bow.bu, MF2, sgn);
	    }
	}
    }
    else {
	// no friction
	bow.Vrel = -bow.bu;
	fric = 0.0;
    }    
}


double BowedString::LagrangeInterp(int xint, double xfrac)
{
    double result;
    // always do cubic unless at end of table, and reference dfric_table where
    // Matlab references grid
    if ((xint == tablesize) || (xint == 0)) {
	/* actually if ((xint+1) == (tablesize+1))... because x is zero-based and
	 * table_size is one smaller than actual size */
	// linear
	result = (1.0 - xfrac)*dfric_table[xint] + xfrac*dfric_table[xint+1];
    }
    else {
	// cubic
	result = (xfrac*(xfrac-1.0)*(xfrac-2.0)/-6.0)*dfric_table[xint-1] +
	    ((xfrac-1.0)*(xfrac+1.0)*(xfrac-2.0)/2.0)*dfric_table[xint] +
	    (xfrac*(xfrac+1.0)*(xfrac-2.0)/-2.0)*dfric_table[xint+1] +
	    (xfrac*(xfrac+1.0)*(xfrac-1.0)/6.0)*dfric_table[xint+2];
    }
    return result;
}


// Vrel and fric are outputs, everything else is an input
void BowedString::Newton_friction(double &Vrel, double &fric, double F, double b,
				  double MF, double sgn)
{
    Vrel = 2.0 * sgn;
    int it = 0;
    double res = 1.0;
    double dfric;
    while (res > tol) {
	if (fabs(Vrel) >= 1.0) {
	    // Vrel_const
	    if (Vrel >= 0.0) fric = 0.35;
	    else fric = -0.35;
	    dfric = 0.0;
	}
	else {
	    // Vrel_var
	    // compute friction using function (not table)
	    double expvel10 = exp(-fabs(Vrel)* 10.0);
	    //double expvel100 = pow(expvel10, 10.0);
	    double expvel100 = expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10;
	    double sgn = Vrel >= 0.0 ? 1.0 : -1.0;
	    fric = sgn * (0.4 * expvel100 + 0.45 * expvel10 + 0.35);
	    dfric = -(40.0 * expvel100 + 4.5 * expvel10);
	}
	
	double R = fric * F;
	double FuncF = Vrel + b + MF * R;
	double Rprime = dfric * F;
	double FuncFprime = 1.0 + MF * Rprime;
	double Vrelnew = Vrel - FuncF / FuncFprime;

	res = fabs(Vrel - Vrelnew);
	Vrel = Vrelnew;
	it++;
	if (it > 30) break;
    }
    
    // final friction computation
    if (fabs(Vrel) >= 1.0) {
	// Vrel_const
	if (Vrel >= 0.0) fric = 0.35;
	else fric = -0.35;
    }
    else {
	// Vrel_var
	// compute friction using function (not table)
	double expvel10 = exp(-fabs(Vrel)* 10.0);
	double expvel100 = expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10*expvel10;
	//double expvel100 = pow(expvel10, 10.0);
	double sgn = Vrel >= 0.0 ? 1.0 : -1.0;
	fric = sgn * (0.4 * expvel100 + 0.45 * expvel10 + 0.35);
    }
}

void BowedString::frictionFingersNeck(double &fric, double &dfric, double velocity)
{
    double coeff = 1.0 / 0.0002;
    fric = 2.0 / M_PI * atan(coeff * velocity);
    dfric = 2.0 / M_PI * coeff / (1.0 + ((velocity*coeff)*(velocity*coeff)));
}

void BowedString::coulombNewton(double &fric, double Fc, double MF2, Bow &bow)
{
    int it = 0;
    double res = 1.0;
    double tol = 1e-13;
    double dfric;
    double mu = 1.5;
    double R, FuncF, Rprime, FuncFprime, Vrelnew;

    if (Fc < 0.0) Fc = 0.0;

    while (res > tol) {
	frictionFingersNeck(fric, dfric, bow.Vrel);
	fric = mu * fric;
	dfric = mu * dfric;
	R = fric * Fc;
	FuncF = bow.Vrel + bow.bu + MF2*R;
	Rprime = dfric * Fc;
	FuncFprime = 1.0 + MF2*Rprime;
	Vrelnew = bow.Vrel - FuncF / FuncFprime;
	res = fabs(bow.Vrel - Vrelnew);
	bow.Vrel = Vrelnew;

	it = it + 1;
	if (it > 1000000) {
	    logMessage(5, "Didn't reach a solution in coulombNewton after 1000000 iterations");
	    break;
	}
    }

    frictionFingersNeck(fric, dfric, bow.Vrel);
    fric = mu * fric;
}


// vector version of coulombNewton
// fric is an output, Vrel is both input and output, other arguments are inputs
// although Fc gets overwritten
// all are vectors of length numFingers except MF2 which is a matrix of size
// numFingers x numFingers
void BowedString::coulombNewton(double *fric, double *Fc, double *MF2,
				double *Vrel, double *bu)
{
    int it = 0;
    double res[MAX_OBJS];
    double tol = 1e-13;
    double dfric[MAX_OBJS];
    double mu = 1.5;
    double R[MAX_OBJS], FuncF[MAX_OBJS], Rprime[MAX_OBJS],
	FuncFprime[MAX_OBJS*MAX_OBJS], Vrelnew[MAX_OBJS];
    double L[MAX_OBJS*MAX_OBJS], U[MAX_OBJS*MAX_OBJS], cy[MAX_OBJS];
    double maxres;
    int i, j;
    
    // Fc is only used as max(Fc, 0) so might as well do it once here
    for (i = 0; i < numFingers; i++) {
	if (Fc[i] < 0.0) Fc[i] = 0.0;
    }

    do {
	for (i = 0; i < numFingers; i++) {
	    frictionFingersNeck(fric[i], dfric[i], Vrel[i]);
	    fric[i] = mu * fric[i];
	    dfric[i] = mu * dfric[i];
	    R[i] = fric[i] * Fc[i];
	    Rprime[i] = dfric[i] * Fc[i];

	}

	// FuncF = Vrel + b + MF2*R
	// can't be merged with first loop as we need all values of R
	for (i = 0; i < numFingers; i++) {
	    FuncF[i] = Vrel[i] + bu[i];
	    for (j = 0; j < numFingers; j++) {
		FuncF[i] += MF2[(j*numFingers)+i] * R[j];
	    }
	}

	// compute FuncFprime matrix
	// FuncFprime = eye + MF2*diagonalRprime
	// post-multiply by diagonal matrix = columns multiplied by Rprime values
	for (i = 0; i < numFingers; i++) {
	    for (j = 0; j < numFingers; j++) {
		FuncFprime[(j*numFingers)+i] = MF2[(j*numFingers)+i] * Rprime[j];
	    }
	    FuncFprime[(i*numFingers)+i] += 1.0;
	}

	// do linear system solve for FuncFprime\FuncF into Vrelnew
	croutDecomposition(FuncFprime, L, U, numFingers);
	croutSolve(L, U, FuncF, Vrelnew, cy, numFingers);

	// update Vrel and res
	for (i = 0; i < numFingers; i++) {
	    Vrelnew[i] = Vrel[i] - Vrelnew[i];
	    res[i] = fabs(Vrel[i] - Vrelnew[i]);
	    Vrel[i] = Vrelnew[i];
	}

	it = it + 1;
	if (it > 1000000) {
	    logMessage(5, "Didn't reach a solution in coulombNewton after 1000000 iterations");
	    break;
	}

	// compute max residual
	maxres = 0.0;
	for (i = 0; i < numFingers; i++) {
	    if (res[i] > maxres) maxres = fabs(res[i]);
	}
    } while (maxres > tol);

    // post-process
    for (i = 0; i < numFingers; i++) {
	frictionFingersNeck(fric[i], dfric[i], Vrel[i]);
	fric[i] = mu * fric[i];
    }
}


double *BowedString::getEnergy()
{
    return energy;
}

void BowedString::setBowParameters(double K, double alpha, double beta,
				   double lambda, double M)
{
    this->KB = K;
    this->alphaB = alpha;
    this->betaB = beta;
    this->lambdaB = lambda;
    this->MB = M;

    logMessage(1, "Set bow parameters to %f %f %f %f %f", KB, alphaB, betaB,
	       lambdaB, MB);
}

void BowedString::setFingerParameters(double Kw, double Ku, double alpha,
				      double beta, double lambda, double M)
{
    KwF = Kw;
    KuF = Ku;
    alphaF = alpha;
    betaF = beta;
    lambdaF = lambda;
    MF = M;

    logMessage(1, "Set finger parameters to %f %f %f %f %f %f", Kw, Ku, alpha,
	       beta, lambda, M);
}

