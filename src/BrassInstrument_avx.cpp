/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * AVX kernels for brass instrument. This is kept in a separate module so that we can
 * compile this file alone with "-mavx", preventing the compiler from issuing AVX
 * instructions elsewhere and ensuring that we can safely use the same binary on
 * non-AVX machines with runtime detection.
 */

#include "BrassInstrument.h"
#include "GlobalSettings.h"
#include "MathUtil.h"

#include <cmath>
using namespace std;

#ifdef USE_AVX
#include <immintrin.h>

#define PMAIN_HIST(ts, loc) (pmainhist[ts][loc])
#define VMAIN_HIST(ts, loc) (vmainhist[ts][loc])

#define PBY_HIST(ts, loc) (pbyhist[ts][loc])
#define VBY_HIST(ts, loc) (vbyhist[ts][loc])

/*
 * AVX accelerated versions of bypass tube pressure and velocity update.
 * These are roughly twice as fast as the plain C versions.
 */
void BrassInstrument::bypassPressureUpdateAVX(int i)
{
    int j, k;
    double val;

    double kk = GlobalSettings::getInstance()->getK();

    // load the constants only once
    __m256d a0d = _mm256_loadu_pd(a0DivRhoCCkk);
    __m256d gammad = _mm256_loadu_pd(gammaM1DivNuCC);
    __m256d b0d = _mm256_loadu_pd(b0);
    __m256d etapid = _mm256_loadu_pd(etaPiDivRho3);
    __m256d invrhod = _mm256_loadu_pd(invRhoCCkk);

    // loop over most values in 4s
    for (j = 0; j < ((bypassSize-2) >> 2); j++) {

	// compute pbyad
	__m256d sbard = _mm256_loadu_pd(&Sbybar1[(j*4)+1]);
	__m256d pbyad = _mm256_div_pd(etapid, sbard);
	pbyad = _mm256_sqrt_pd(pbyad);
	pbyad = _mm256_mul_pd(pbyad, gammad);
	__m256d yd = pbyad;
	pbyad = _mm256_mul_pd(pbyad, b0d);
	pbyad = _mm256_add_pd(pbyad, a0d);
	__m256d tempbard = _mm256_loadu_pd(&tempbybar1[(j*4)+1]);
	pbyad = _mm256_div_pd(tempbard, pbyad);

	// now pbybd
	__m256d pbybd = _mm256_mul_pd(invrhod, pbyad);

	// pbycd
	__m256d pbycd = _mm256_mul_pd(pbyad, yd);

	// pbydd - inverse of the version used in C! and multiplied by bya
	__m256d pbydd = _mm256_loadu_pd(&hbyvec[(j*4)+1]);
	pbydd = _mm256_mul_pd(sbard, pbydd);
	pbydd = _mm256_div_pd(pbyad, pbydd);

	__m256d pbyd1d = _mm256_loadu_pd(&tempby1[j*4]);
	__m256d sbyd = _mm256_loadu_pd(&Sby1[j*4]);
	pbyd1d = _mm256_mul_pd(pbyd1d, sbyd);
	pbyd1d = _mm256_mul_pd(pbyd1d, pbydd);

	__m256d pbyd2d = _mm256_loadu_pd(&tempby1[(j*4)+1]);
	sbyd = _mm256_loadu_pd(&Sby1[(j*4)+1]);
	pbyd2d = _mm256_mul_pd(pbyd2d, sbyd);
	pbyd2d = _mm256_mul_pd(pbyd2d, pbydd);

	__m256d vald = _mm256_setzero_pd();

	for (k = 0; k < historySize; k++) {
	    // load filter co-efficients
	    __m256d Adiffd = _mm256_load_pd(&Adiff4[k*4]);
	    __m256d Bsumd = _mm256_load_pd(&Bsum4[k*4]);
	    __m256d Ad = _mm256_load_pd(&A4[k*4]);

	    // do PBY_HIST terms
	    __m256d td1 = _mm256_mul_pd(pbybd, Adiffd);
	    __m256d td2 = _mm256_mul_pd(pbycd, Bsumd);
	    td1 = _mm256_sub_pd(td1, td2);
	    td2 = _mm256_loadu_pd(&pbyhist[k][(j*4)+1]);
	    td1 = _mm256_mul_pd(td1, td2);
	    vald = _mm256_add_pd(vald, td1);

	    // VBY_HIST terms
	    td1 = _mm256_mul_pd(pbyd1d, Ad);
	    td2 = _mm256_loadu_pd(&vbyhist[k][(j*4)]);
	    td1 = _mm256_mul_pd(td1, td2);
	    vald = _mm256_add_pd(vald, td1);

	    td1 = _mm256_mul_pd(pbyd2d, Ad);
	    td2 = _mm256_loadu_pd(&vbyhist[k][(j*4)+1]);
	    td1 = _mm256_mul_pd(td1, td2);
	    vald = _mm256_sub_pd(vald, td1);
	}

	// store result
	_mm256_storeu_pd(&pby[(j*4)+1], vald);
    }

    // do remainder
    for (j = 1 + (((bypassSize-2)>>2)<<2); j < (bypassSize - 1); j++) {
	double bya, byb, byc, byd, byd1, byd2;

	bya = tempbybar1[j] /
	    (A[0]/(rho*c*c*kk) + (gamma-1.0)*sqrt(eta*M_PI/(rho*rho*rho*Sbybar1[j]))*B[0] / (nu*c*c));
	byb = bya / (rho*c*c*kk);
	byc = -bya * ((gamma-1.0)*sqrt(eta*M_PI/(rho*rho*rho*Sbybar1[j])) / (nu*c*c));
	byd = hbyvec[j] * Sbybar1[j];
	byd1 = bya * tempby1[j-1] * Sby1[j-1] / byd;
	byd2 = -bya * tempby1[j] * Sby1[j] / byd;

	val = 0.0;

	for (k = 0; k < historySize; k++) {
	    val += byb * (Adiff[k] * PBY_HIST(k, j));
	    val += byc * (Bsum[k]  * PBY_HIST(k, j));
	    val += byd1 * (A[k] * VBY_HIST(k, j-1));
	    val += byd2 * (A[k] * VBY_HIST(k, j));
	}

	pby[j] = val;	
    }
}

void BrassInstrument::bypassVelocityUpdateAVX(int i)
{
    int j, k;
    double val;

    double kk = GlobalSettings::getInstance()->getK();

    // load constants
    __m256d a0d = _mm256_load_pd(&A4[0]);
    __m256d b0d = _mm256_loadu_pd(b0);
    __m256d rhokkd = _mm256_loadu_pd(rhoDivKk);
    __m256d onep5d = _mm256_loadu_pd(oneP5EtaPi);
    __m256d rhoetad = _mm256_loadu_pd(rhoEtaPi);


    // loop over most values in 4s
    for (j = 0; j < ((bypassSize-1) >> 2); j++) {

	// compute vbyad first
	__m256d sby1d = _mm256_loadu_pd(&Sby1[j*4]);
	__m256d yd = _mm256_div_pd(rhoetad, sby1d);
	yd = _mm256_sqrt_pd(yd); // this term needed for vbyad and vbydd
	__m256d zd = _mm256_div_pd(onep5d, sby1d); // this one needed for vbyad and vbycd
	__m256d tmp1 = _mm256_mul_pd(yd, b0d);
	__m256d tmp2 = _mm256_add_pd(zd, rhokkd);
	tmp2 = _mm256_mul_pd(a0d, tmp2);
	tmp2 = _mm256_add_pd(tmp2, tmp1); // got denominator now
	__m256d vbyad = _mm256_loadu_pd(&tempby1[j*4]);
	vbyad = _mm256_div_pd(vbyad, tmp2);
	
	// now get vbybd
	__m256d vbybd = _mm256_mul_pd(vbyad, rhokkd);

	// vbycd
	__m256d vbycd = _mm256_mul_pd(vbyad, zd);

	// vbydd
	__m256d vbydd = _mm256_mul_pd(vbyad, yd);

	// vbyed
	__m256d vbyed = _mm256_loadu_pd(&hbyvec[j*4]);
	vbyed = _mm256_div_pd(vbyad, vbyed);

	// load filter co-efficients
	__m256d Adiffd = _mm256_load_pd(&Adiff4[0]);
	__m256d Asumd = _mm256_load_pd(&Asum4[0]);
	__m256d Bsumd = _mm256_load_pd(&Bsum4[0]);
	__m256d Ad = _mm256_load_pd(&A4[0]);

	// do first step using pby
	__m256d vald = _mm256_loadu_pd(&vbyhist[0][j*4]);
	__m256d td1 = _mm256_mul_pd(vbybd, Adiffd);
	__m256d td2 = _mm256_mul_pd(vbycd, Asumd);
	td1 = _mm256_sub_pd(td1, td2);
	td2 = _mm256_mul_pd(vbydd, Bsumd);
	td1 = _mm256_sub_pd(td1, td2);
	vald = _mm256_mul_pd(vald, td1); // compute all the VBY_HIST terms

	td1 = _mm256_loadu_pd(&pby[j*4]);
	td2 = _mm256_loadu_pd(&pby[(j*4)+1]);
	td1 = _mm256_sub_pd(td1, td2);
	td1 = _mm256_mul_pd(td1, Ad);
	td1 = _mm256_mul_pd(td1, vbyed);
	vald = _mm256_add_pd(vald, td1);
	
	// iterate over other steps
	for (k = 1; k < historySize; k++) {
	    __m256d vbyhistd = _mm256_loadu_pd(&vbyhist[k][j*4]);

	    // load filter co-efficients
	    Adiffd = _mm256_load_pd(&Adiff4[k*4]);
	    Asumd = _mm256_load_pd(&Asum4[k*4]);
	    Bsumd = _mm256_load_pd(&Bsum4[k*4]);
	    Ad = _mm256_load_pd(&A4[k*4]);

	    // do vby terms
	    td1 = _mm256_mul_pd(vbybd, Adiffd);
	    td2 = _mm256_mul_pd(vbycd, Asumd);
	    td1 = _mm256_sub_pd(td1, td2);
	    td2 = _mm256_mul_pd(vbydd, Bsumd);
	    td1 = _mm256_sub_pd(td1, td2);
	    td1 = _mm256_mul_pd(td1, vbyhistd);
	    vald = _mm256_add_pd(vald, td1);

	    // do pby term
	    td1 = _mm256_loadu_pd(&pbyhist[k-1][j*4]);
	    td2 = _mm256_loadu_pd(&pbyhist[k-1][(j*4)+1]);
	    td1 = _mm256_sub_pd(td1, td2);
	    td1 = _mm256_mul_pd(Ad, td1);
	    td1 = _mm256_mul_pd(vbyed, td1);
	    vald = _mm256_add_pd(vald, td1);
	}

	// store val
	_mm256_storeu_pd(&vby[j*4], vald);
    }

    // do odd elements
    for (j = (((bypassSize-1)>>2)<<2); j < (bypassSize-1); j++) {
	double bya, byb, byc, byd, bye;
    
	bya = tempby1[j] /
	    (A[0] * (rho/kk + 1.5*eta*M_PI/Sby1[j]) + B[0]*sqrt(rho*eta*M_PI/Sby1[j]));
	byb = bya * (rho / kk);
	byc = -bya * (1.5*eta*M_PI / Sby1[j]);
	byd = -bya * sqrt(rho*eta*M_PI / Sby1[j]);
	bye = bya / hbyvec[j];

	val = byb * (Adiff[0] * VBY_HIST(0, j));
	val += byc * (Asum[0]  * VBY_HIST(0, j));
	val += byd * (Bsum[0]  * VBY_HIST(0, j));
	val += bye * A[0] * (pby[j] - pby[j+1]);

	for (k = 1; k < historySize; k++) {
	    val += byb * (Adiff[k] * VBY_HIST(k, j));
	    val += byc * (Asum[k]  * VBY_HIST(k, j));
	    val += byd * (Bsum[k]  * VBY_HIST(k, j));
	    val += bye * A[k] * (PBY_HIST(k-1, j) - PBY_HIST(k-1, j+1));
	}

	vby[j] = val;
    }
    {
	double bya, byb, byc, byd, bye;
    
	bya = tempby1[j] /
	    (A[0] * (rho/kk + 1.5*eta*M_PI/Sby1[j]) + B[0]*sqrt(rho*eta*M_PI/Sby1[j]));
	byb = bya * (rho / kk);
	byc = -bya * (1.5*eta*M_PI / Sby1[j]);
	byd = -bya * sqrt(rho*eta*M_PI / Sby1[j]);

	val = 0.0;
	for (k = 0; k < historySize; k++) {
	    val += byb * (Adiff[k] * VBY_HIST(k, j));
	    val += byc * (Asum[k]  * VBY_HIST(k, j));
	    val += byd * (Bsum[k]  * VBY_HIST(k, j));
	}
	vby[j] = val;
    }
}

/*
 * AVX accelerated versions of main tube pressure and velocity update.
 * These are roughly 3 times faster than the plain C versions.
 */
void BrassInstrument::mainVelocityUpdateAVX()
{
    int j, k;
    double val;

    // loop over most values in 4s
    for (j = 0; j < ((mainSize-1) >> 2); j++) {
	// load values from j-based arrays (vmain*)
	__m256d vmainbd = _mm256_load_pd(&vmainb[j*4]);
	__m256d vmaincd = _mm256_load_pd(&vmainc[j*4]);
	__m256d vmaindd = _mm256_load_pd(&vmaind[j*4]);
	__m256d vmained = _mm256_load_pd(&vmaine[j*4]);

	// load filter co-efficients
	__m256d Adiffd = _mm256_load_pd(&Adiff4[0]);
	__m256d Asumd = _mm256_load_pd(&Asum4[0]);
	__m256d Bsumd = _mm256_load_pd(&Bsum4[0]);
	__m256d Ad = _mm256_load_pd(&A4[0]);

	// do first step using pmain
	__m256d vald = _mm256_loadu_pd(&vmainhist[0][j*4]);
	__m256d td1 = _mm256_mul_pd(vmainbd, Adiffd);
	__m256d td2 = _mm256_mul_pd(vmaincd, Asumd);
	td1 = _mm256_add_pd(td1, td2);
	td2 = _mm256_mul_pd(vmaindd, Bsumd);
	td1 = _mm256_add_pd(td1, td2);
	vald = _mm256_mul_pd(vald, td1); // compute all the VMAIN_HIST terms

	td1 = _mm256_loadu_pd(&pmain[j*4]);
	td2 = _mm256_loadu_pd(&pmain[(j*4)+1]);
	td1 = _mm256_sub_pd(td1, td2);
	td1 = _mm256_mul_pd(td1, Ad);
	td1 = _mm256_mul_pd(td1, vmained);
	vald = _mm256_add_pd(vald, td1);

	// iterate over other steps
	for (k = 1; k < historySize; k++) {
	    __m256d vmainhistd = _mm256_loadu_pd(&vmainhist[k][j*4]);

	    // load filter co-efficients
	    Adiffd = _mm256_load_pd(&Adiff4[k*4]);
	    Asumd = _mm256_load_pd(&Asum4[k*4]);
	    Bsumd = _mm256_load_pd(&Bsum4[k*4]);
	    Ad = _mm256_load_pd(&A4[k*4]);

	    // do vmain terms
	    td1 = _mm256_mul_pd(vmainbd, Adiffd);
	    td2 = _mm256_mul_pd(vmaincd, Asumd);
	    td1 = _mm256_add_pd(td1, td2);
	    td2 = _mm256_mul_pd(vmaindd, Bsumd);
	    td1 = _mm256_add_pd(td1, td2);
	    td1 = _mm256_mul_pd(td1, vmainhistd);
	    vald = _mm256_add_pd(vald, td1);

	    // do pmain term
	    td1 = _mm256_loadu_pd(&pmainhist[k-1][j*4]);
	    td2 = _mm256_loadu_pd(&pmainhist[k-1][(j*4)+1]);
	    td1 = _mm256_sub_pd(td1, td2);
	    td1 = _mm256_mul_pd(Ad, td1);
	    td1 = _mm256_mul_pd(vmained, td1);
	    vald = _mm256_add_pd(vald, td1);
	}

	// store val
	_mm256_storeu_pd(&vmain[j*4], vald);
    }

    // do odd elements
    for (j = (((mainSize-1)>>2)<<2); j < (mainSize-1); j++) {
	val = vmainb[j] * (Adiff[0] * VMAIN_HIST(0, j));
	val += vmainc[j] * (Asum[0]  * VMAIN_HIST(0, j));
	val += vmaind[j] * (Bsum[0]  * VMAIN_HIST(0, j));
	val += vmaine[j] * A[0] * (pmain[j] - pmain[j+1]);

	for (k = 1; k < historySize; k++) {
	    val += vmainb[j] * (Adiff[k] * VMAIN_HIST(k, j));
	    val += vmainc[j] * (Asum[k]  * VMAIN_HIST(k, j));
	    val += vmaind[j] * (Bsum[k]  * VMAIN_HIST(k, j));
	    val += vmaine[j] * A[k] * (PMAIN_HIST(k-1, j) - PMAIN_HIST(k-1, j+1));
	}

	vmain[j] = val;
    }
    /* final entry is different. j is still used here, it's now mainSize-1 */
    val = 0.0;
    for (k = 0; k < historySize; k++) {
	val += vmainb[j] * (Adiff[k] * VMAIN_HIST(k, j));
	val += vmainc[j] * (Asum[k]  * VMAIN_HIST(k, j));
	val += vmaind[j] * (Bsum[k]  * VMAIN_HIST(k, j));
    }
    vmain[j] = val;
}

void BrassInstrument::mainPressureUpdateAVX()
{
    int j, k;
    double val;

    // loop over values in 4s
    for (j = 0; j < ((mainSize-2) >> 2); j++) {
	__m256d vald = _mm256_setzero_pd();
	
	// load values from j-based arrays
	__m256d pmainad = _mm256_load_pd(&pmaina[j*4]);
	__m256d pmainbd = _mm256_load_pd(&pmainb[j*4]);
	__m256d pmaincd = _mm256_load_pd(&pmainc[j*4]);

	__m256d Smaind1 = _mm256_load_pd(&Smain[j*4]);
	__m256d Smaind2 = _mm256_loadu_pd(&Smain[(j*4)+1]);
	// pre-multiply these by pmainc
	Smaind1 = _mm256_mul_pd(Smaind1, pmaincd);
	Smaind2 = _mm256_mul_pd(Smaind2, pmaincd);

	for (k = 0; k < historySize; k++) {
	    __m256d pmainhistd = _mm256_loadu_pd(&pmainhist[k][(j*4)+1]);
	    
	    __m256d vmainhistd1 = _mm256_load_pd(&vmainhist[k][(j*4)]);
	    __m256d vmainhistd2 = _mm256_loadu_pd(&vmainhist[k][(j*4)+1]);

	    // load filter co-efficients
	    __m256d Adiffd = _mm256_load_pd(&Adiff4[k*4]);
	    __m256d Bsumd = _mm256_load_pd(&Bsum4[k*4]);
	    __m256d Ad = _mm256_load_pd(&A4[k*4]);

	    // update vald
	    __m256d td1 = _mm256_mul_pd(pmainad, Adiffd);
	    __m256d td2 = _mm256_mul_pd(pmainbd, Bsumd);
	    td1 = _mm256_add_pd(td1, td2);
	    td1 = _mm256_mul_pd(td1, pmainhistd);
	    vald = _mm256_add_pd(vald, td1);

	    td1 = _mm256_mul_pd(Smaind1, Ad);
	    td1 = _mm256_mul_pd(td1, vmainhistd1);
	    vald = _mm256_add_pd(vald, td1);

	    td1 = _mm256_mul_pd(Smaind2, Ad);
	    td1 = _mm256_mul_pd(td1, vmainhistd2);
	    vald = _mm256_sub_pd(vald, td1);
	}

	_mm256_storeu_pd(&pmain[(j*4)+1], vald);
    }


    // do remainder of elements
    for (j = 1 + (((mainSize-2)>>2)<<2); j < (mainSize - 1); j++) {
	val = 0.0;

	for (k = 0; k < historySize; k++) {
	    val += pmaina[j-1] * (Adiff[k] * PMAIN_HIST(k, j));
	    val += pmainb[j-1] * (Bsum[k]  * PMAIN_HIST(k, j));
	    val += pmainc[j-1] * Smain[j-1] * (A[k] * VMAIN_HIST(k, j-1));
	    val -= pmainc[j-1] * Smain[j] * (A[k] * VMAIN_HIST(k, j));
	}

	pmain[j] = val;
    }
}

/*
 * There's probably no point in AVX accelerating the default tube
 * updates as these are normally very small and take hardly any time
 * anyway.
 */

#endif
