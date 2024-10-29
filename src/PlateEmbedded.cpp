/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

/*
 * This is a very complex class that can do lots of different things. It can
 * simulate linear and non-linear plates, circular and rectangular, embedded
 * within a box of air. Despite its name, it can also handle membranes, and non-
 * embedded plates.
 *
 * A lot of the complexity comes from the fact that it's highly optimised. Most
 * of the critical matrices are stored in banded format (3 diagonal bands of 3
 * values each, or 5 bands of 5 values each) instead of the standard CSR.
 * Additionally, a number of the operations now use optimised custom code instead
 * of sequences of standard operations: these are computeSYMASY (which decomposes
 * BigMat into symmetric and asymmetric parts), computeBigMat (mostly a matrix-by-
 * matrix multiplication, but also adds in the Laplacian, DDDD_F, at the same
 * time), computeLOPWF (which computes LOPW and LOPF, both of which are the sums
 * of several vector-by-matrix products), and computeTemp1_6 (which computes 6
 * matrix-by-vector products at once).
 *
 * The startup code uses some banded matrices and some CSR, with quite a lot of
 * conversion back and forth. This is because the CSR library has a much richer
 * set of functions available, but some operations (notably matrix-matrix additions
 * and multiplications) are so much faster in banded form that it's worth
 * converting the matrices to banded just to do them. All of the matrices that will
 * be used in the main loop are converted to banded form in the initialisation
 * phase.
 *
 * If the plate/membrane is embedded in an airbox, it will be associated with an
 * Embedding object, which binds the plate and airbox together and handles the
 * interface between them. If this is the case, setInterpolation will be called
 * after the PlateEmbedded is created but before any timesteps are run. The two
 * interpolation matrices (IMat and JMat) are passed in here, along with a buffer
 * used to pass data to and from the Embedding.
 *
 * The non-linear version of the timestep update requires a linear system solve
 * operation. This is implemented in pcgSolve5x5 which performs the preconditioned
 * conjugate gradient algorithm on the supplied banded 5x5 system matrix. The
 * preconditioner is a triangular solve on the Cholesky decomposition of DDDD_F.
 * This preconditioner makes parallelisation very difficult and is the reason that
 * there's no GPU version of this. However, it should be possible to create a
 * reasonable GPU version for the linear case.
 *
 * Notes:
 *  - for rectangular plates, most of the matrices etc. are created as if the plate
 *    had an additional "halo" of points around the outside. The rows and columns
 *    related to these extra points are removed during start-up, by calling
 *    CSR_cut_cols and CSR_cut_rows. The TruePhi array lists the indices that are
 *    not to be removed.
 *
 *  - for circular plates, there are no extra rows and columns. Everything is
 *    mostly set up as for a rectangular plate. The points on the plate surface
 *    that fall outside the circle are not used, but (unlike in the original Matlab
 *    version) they do still have storage allocated for them! This makes the layout
 *    of the matrices more regular and allows some optimisations that otherwise
 *    couldn't be done. The CSR_zero_cols and CSR_zero_rows functions are used to
 *    set matrix entries relating to these unused points to zero. Again TruePhi
 *    contains a list of the points that are actually to be used.
 *
 *  - a few options available in the original Matlab versions of the codes (MP3D
 *    and bass drum) were never ported to this version: vent holes, anti-aliased
 *    interpolation, and certain boundary conditions.
 */

#include "PlateEmbedded.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"
#include "Logger.h"

#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;

#ifdef USE_SSE
#define m3x3_multiply m3x3_vector_SSE
#define m5x5_multiply m5x5_vector_SSE
#else
#define m3x3_multiply m3x3_vector_mult
#define m5x5_multiply m5x5_vector_mult
#endif

static const char *profileNames[] = {
    "B and C (linear)",
    "Interpolation 1 (linear)",
    "Inputs (linear)",
    "Interpolation loop (linear)",

    "Compute temp1-6",
    "Compute LOPW/F",
    "Compute aa",
    "Inputs (non-linear)",
    "Compute bb",
    "Compute BigMat",
    "Compute SYM and ASY",
    "Compute Right",
    "Compute BigRight",
    "PCG solve",
    "Update u"
};

PlateEmbedded::PlateEmbedded(string name, Material *material, double thickness, double tension,
			     double lx, double ly, double t60_0, double sig1, double cx, double cy,
			     double cz, bool isMembrane, bool isCircular)
    : Component2D(name)
{
    init(material->getPoissonsRatio(), material->getDensity(), material->getYoungsModulus(), thickness,
	 tension, lx, ly, t60_0, sig1, cx, cy, cz, isMembrane, isCircular);
}

PlateEmbedded::PlateEmbedded(string name, double nu, double rho, double E, double thickness,
			     double tension, double lx, double ly, double t60_0, double sig1, double cx, double cy,
			     double cz, bool isMembrane, bool isCircular)
    : Component2D(name)
{
    init(nu, rho, E, thickness, tension, lx, ly, t60_0, sig1, cx, cy, cz, isMembrane,
	 isCircular);
}

PlateEmbedded::~PlateEmbedded()
{
    delete[] TruePhi;

    CSR_free(B);
    CSR_free(C);
    CSR_free(Bw);

    delete[] Right;

    delete[] vtemp;

    if (energy) {
	delete[] energy;
	delete[] energytmp;
	delete[] energytmp2;
	CSR_free(DD);
	if (!linear) {
	    CSR_free(DD_F);
	}
	CSR_free(Dxp);
	CSR_free(Dyp);
    }

    if (!linear) {
#ifdef USE_SSE
	pcgFreeSSE(pcg);
#else
	pcgFreeCholesky(pcg);
#endif

	delete[] Phi;
	delete[] Phi1;
	delete[] Phi0;

	CSR_free(DDDD_F);
	CSR_free(LOPW);
	CSR_free(LOPF);
	CSR_free(BigMat);
	
	delete[] aa;
	delete[] bb;

	delete[] temp1;
	delete[] temp2;
	delete[] temp3;
	delete[] temp4;
	delete[] temp5;
	delete[] temp6;

	delete[] BigRight;

	free5x5(BigMat5x5);

	free3x3(LOPW3x3);
	free3x3(LOPF3x3);

	free3x3(DLMM3);
	free3x3(DLMP3);
	free3x3(DLPM3);
	free3x3(DLPP3);
	free3x3(Dxxw3);
	free3x3(Dyyw3);
	free3x3(Dxx_F3);
	free3x3(Dyy_F3);
	
	free5x5(SYM);
	free5x5(ASY);
    }

    if (Diff_tmp1) {
	delete[] Diff_tmp1;
    }

    if (cplfacIMatJMat3x3) {
	free3x3(cplfacIMatJMat3x3);
    }
}

void PlateEmbedded::swapBuffers(int n)
{
    double *tmp;

    // do standard state swap and state record if on
    Component::swapBuffers(n);

    if (!linear) {
	// swap Phi buffers
	tmp = Phi0;
	Phi0 = Phi1;
	Phi1 = Phi;
	Phi = tmp;

	// remember to preserve Phi and u
	memcpy(Phi, Phi1, ss * sizeof(double));
    }
    memcpy(u, u1, ss * sizeof(double));
}

void PlateEmbedded::logMatrices()
{
    saveMatrix(B, "B");
    saveMatrix(C, "C");
    saveMatrix(Bw, "Bw");

    if (!linear) {
	saveMatrix(DDDD_F, "DDDD_F");
	saveMatrix(IMat, "IMat");
	saveMatrix(JMat, "JMat");
	
	saveMatrix3x3(DLMM3, "DLMM");
	saveMatrix3x3(DLMP3, "DLMP");
	saveMatrix3x3(DLPM3, "DLPM");
	saveMatrix3x3(DLPP3, "DLPP");
	saveMatrix3x3(Dxxw3, "Dxxw");
	saveMatrix3x3(Dyyw3, "Dyyw");
	saveMatrix3x3(Dxx_F3, "Dxx_F");
	saveMatrix3x3(Dyy_F3, "Dyy_F");
    }
}

int PlateEmbedded::getGPUScore()
{
    // no GPU acceleration due to linear system solve
    return GPU_SCORE_NO;
}



/*==============================================================================
 *
 * Main timestep functions
 *
 *============================================================================*/
void PlateEmbedded::runTimestep(int n)
{
    int i;
    int II;

    if (!interpBuffer) {
	/* non-embedded */
	if (linear) {
	    profiler->start(0);
	    CSR_matrix_vector_mult(B, u1, u);
	    CSR_matrix_vector(C, u2, u, 0, 2);
	    profiler->end(0);
	}
	else {
	    profiler->start(4);
	    computeTemp1_6();
	    profiler->end(4);

	    profiler->start(5);
	    computeLOPWF();
	    profiler->end(5);

	    /* calculate aa */
	    /* aa = B*u1-C*u2+Bw*(u1-u2)+LOPF*Phi0
	       /* our Bw is now factored into B and C */
	    profiler->start(6);
	    CSR_matrix_vector_mult(B, u1, aa);
	    CSR_matrix_vector(C, u2, aa, 0, 2);
	    m3x3_multiply(LOPF3x3, Phi0, vtemp);
	    for (i = 0; i < ss; i++) {
		aa[i] += vtemp[i];
	    }
	    profiler->end(6);

	    // FIXME: not sure about u1 and u2 here, but not sure what else we could
	    // use instead... most inputs don't use them anyway
	    profiler->start(7);
	    runInputs(n, aa, u1, u2);
	    profiler->end(7);

	    // calculate bb
	    // bb = -(DDDD_F * Phi1)
	    profiler->start(8);
	    CSR_matrix_vector_mult(DDDD_F, Phi1, bb);
	    for (i = 0; i < ss; i++) {
		bb[i] = -bb[i];
	    }
	    profiler->end(8);

	    profiler->start(9);
	    computeBigMat();
	    profiler->end(9);

	    if (symmetric) {
		profiler->start(10);
		computeSYMASY();
		profiler->end(10);
	    }

	    // calculate BigRight
	    // BigRight = bb - LOPW*aa [- ASY*Phi]
	    profiler->start(12);
	    m3x3_multiply(LOPW3x3, aa, BigRight);
	    if (symmetric) {
		m5x5_multiply(ASY, Phi, vtemp);
		for (i = 0; i < ss; i++) {
		    BigRight[i] = bb[i] - BigRight[i] - vtemp[i];
		}
		profiler->end(12);

		// solve to get Phi
		profiler->start(13);
		pcgSolve5x5(pcg, SYM, Phi, BigRight);
		profiler->end(13);
	    }
	    else {
		for (i = 0; i < ss; i++) {
		    BigRight[i] = bb[i] - BigRight[i];
		}
		profiler->end(12);

		// solve to get Phi
		profiler->start(13);
		pcgSolve5x5(pcg, BigMat5x5, Phi, BigRight);
		profiler->end(13);
	    }

	    // update u based on Phi
	    // u = aa + LOPF * Phi
	    profiler->start(14);
	    m3x3_multiply(LOPF3x3, Phi, u);
	    for (i = 0; i < ss; i++) {
		u[i] = u[i] + aa[i];
	    }
	    profiler->end(14);
	}
    }
    else {

	/*
	 * Standard embedded case
	 */
	if (linear) {
	    /* compute Right */
	    /* Right = B*u1-C*u0+Bw*(u1-u2)+cplfac*IMat*(JMat*u2)+...
	       Bf*IMat*(Diff_np+2*gamma^2*Diff_n-Diff_nm); */
	    /* our Bw is now factored into B and C */
	    // embedding should have left the Bf*Imat... term in interpBuffer for us
	    profiler->start(0);
	    CSR_matrix_vector_mult(C, u2, Right);
	    for (i = 0; i < ss; i++) {
		Right[i] = interpBuffer[i] - Right[i];
	    }
	    CSR_matrix_vector(B, u1, Right, 0, 1);
	    profiler->end(0);

	    profiler->start(1);
	    if (cplfacIMatJMat3x3 != NULL) {
		m3x3_multiply(cplfacIMatJMat3x3, u2, vtemp);
	    }
	    else {
		CSR_matrix_vector_mult(JMat, u2, Diff_tmp1);
		applyIMat(Diff_tmp1, vtemp, cplfac);
	    }
	    for (i = 0; i < ss; i++) {
		Right[i] += vtemp[i];
	    }
	    profiler->end(1);

	    // FIXME: not sure about u1 and u2 here, but not sure what else we could
	    // use instead... most inputs don't use them anyway
	    profiler->start(2);
	    runInputs(n, Right, u1, u2);
	    profiler->end(2);

	    profiler->start(3);
	    memcpy(u, Right, ss * sizeof(double));
	    for (II = 0; II < iterinv; II++) {
		if (cplfacIMatJMat3x3 != NULL) {
		    m3x3_multiply(cplfacIMatJMat3x3, u, vtemp);
		}
		else {
		    CSR_matrix_vector_mult(JMat, u, Diff_tmp1);
		    applyIMat(Diff_tmp1, vtemp, cplfac);
		}
		for (i = 0; i < ss; i++) {
		    u[i] = Right[i] - vtemp[i];
		}
	    }
	    profiler->end(3);
	}
	else {
	    // non-linear case
	    profiler->start(4);
	    computeTemp1_6();
	    profiler->end(4);

	    profiler->start(5);
	    computeLOPWF();
	    profiler->end(5);

	    /* calculate aa */
	    /* aa = B*u1-C*u2+Bw*(u1-u2)+LOPF*Phi0+cplfac*IMat*(JMat*u2)+...
	       Bf*IMat*(Diff_np+2*gamma^2*Diff_n-Diff_nm); */
	    /* our Bw is now factored into B and C */
	    // the Bf*IMat... term is already in interpBuffer
	    profiler->start(6);
	    CSR_matrix_vector_mult(C, u2, aa);
	    for (i = 0; i < ss; i++) {
		aa[i] = interpBuffer[i] - aa[i];
	    }
	    CSR_matrix_vector(B, u1, aa, 0, 1);

	    m3x3_multiply(LOPF3x3, Phi0, vtemp);
	    for (i = 0; i < ss; i++) {
		aa[i] += vtemp[i];
	    }

	    if (cplfacIMatJMat3x3 != NULL) {
		m3x3_multiply(cplfacIMatJMat3x3, u2, vtemp);
	    }
	    else {
		CSR_matrix_vector_mult(JMat, u2, Diff_tmp1);
		applyIMat(Diff_tmp1, vtemp, cplfac);
	    }
	    for (i = 0; i < ss; i++) {
		aa[i] += vtemp[i];
	    }
	    profiler->end(6);

	    // FIXME: not sure about u1 and u2 here, but not sure what else we could
	    // use instead... most inputs don't use them anyway
	    profiler->start(7);
	    runInputs(n, aa, u1, u2);
	    profiler->end(7);
	
	    // calculate bb
	    // bb = -(DDDD_F * Phi1)
	    profiler->start(8);
	    CSR_matrix_vector_mult(DDDD_F, Phi1, bb);
	    for (i = 0; i < ss; i++) {
		bb[i] = -bb[i];
	    }
	    profiler->end(8);

	    profiler->start(9);
	    computeBigMat();
	    profiler->end(9);

	    if (symmetric) {
		profiler->start(10);
		computeSYMASY();
		profiler->end(10);
	    }

	    for (II = 0; II < iterinv; II++) {
		// calculate Right
		// Right = aa - cplfac*IMat*(JMat*u)
		profiler->start(11);
		if (cplfacIMatJMat3x3 != NULL) {
		    m3x3_multiply(cplfacIMatJMat3x3, u, Right);
		}
		else {
		    CSR_matrix_vector_mult(JMat, u, Diff_tmp1);
		    applyIMat(Diff_tmp1, Right, cplfac);
		}
		for (i = 0; i < ss; i++) {
		    Right[i] = aa[i] - Right[i];
		}
		profiler->end(11);

		// calculate BigRight
		// BigRight = bb - LOPW*Right [- ASY*Phi]
		profiler->start(12);
		m3x3_multiply(LOPW3x3, Right, BigRight);
		if (symmetric) {
		    m5x5_multiply(ASY, Phi, vtemp);
		    for (i = 0; i < ss; i++) {
			BigRight[i] = bb[i] - BigRight[i] - vtemp[i];
		    }
		    profiler->end(12);

		    // solve to get Phi
		    profiler->start(13);
		    pcgSolve5x5(pcg, SYM, Phi, BigRight);
		    profiler->end(13);
		}
		else {
		    for (i = 0; i < ss; i++) {
			BigRight[i] = bb[i] - BigRight[i];
		    }
		    profiler->end(12);

		    // solve to get Phi
		    profiler->start(13);
		    pcgSolve5x5(pcg, BigMat5x5, Phi, BigRight);
		    profiler->end(13);
		}

		// update u based on Phi
		// u = Right + LOPF * Phi
		profiler->start(14);
		m3x3_multiply(LOPF3x3, Phi, u);
		for (i = 0; i < ss; i++) {
		    u[i] = u[i] + Right[i];
		}
		profiler->end(14);
	    }
	}

	// copy u - u2 to the interpolation buffer for the embedding
	for (i = 0; i < ss; i++) {
	    interpBuffer[i] = u[i] - u2[i];
	}
    }

    if (energy) {
	double SR = GlobalSettings::getInstance()->getSampleRate();
	double etot = 0.0;
	double kin = 0.0;
	for (i = 0; i < ss; i++) {
	    double v1 = SR * (u[i] - u1[i]);
	    kin += v1*v1;
	}
	kin *= 0.5 * rho * H;

	double pot = 0.0;
	CSR_matrix_vector_mult(DD, u, energytmp);
	CSR_matrix_vector_mult(DD, u1, energytmp2);
	for (i = 0; i < DD->nrow; i++) {
	    pot += (energytmp[i] * energytmp2[i]);
	}
	pot *= 0.5 * D;

	double potme = 0.0;
	if (isMembrane) {
	    CSR_matrix_vector_mult(Dxp, u1, energytmp);
	    CSR_matrix_vector_mult(Dxp, u, energytmp2);
	    for (i = 0; i < Dxp->nrow; i++) {
		potme += 0.5 * T * energytmp[i] * energytmp2[i];
	    }
	    CSR_matrix_vector_mult(Dyp, u1, energytmp);
	    CSR_matrix_vector_mult(Dyp, u, energytmp2);
	    for (i = 0; i < Dyp->nrow; i++) {
		potme += 0.5 * T * energytmp[i] * energytmp2[i];
	    }
	}

	double nlen = 0.0;
	if (!linear) {
	    CSR_matrix_vector_mult(DD_F, Phi, energytmp);
	    CSR_matrix_vector_mult(DD_F, Phi1, energytmp2);
	    for (i = 0; i < DD_F->nrow; i++) {
		nlen += (energytmp[i]*energytmp[i]) + (energytmp2[i]*energytmp2[i]);
	    }
	    nlen *= 1.0 / (4.0*E*H);
	}

	etot = h*h*pot + h*h*kin + h*h*nlen + h*h*potme;

	energy[n] = etot;
    }
}

#define DO_SYM_ASY_ENTRY \
	val = BigMat5x5->values[idx]; \
	valT = BigMat5x5->values[idxT]; \
	SYM->values[idx] = 0.5 * (val + valT); \
	SYM->values[idxT] = 0.5 * (val + valT); \
	ASY->values[idx] = 0.5 * (val - valT); \
	ASY->values[idxT] = 0.5 * (valT - val); \
	idx++


/*
 * Implements:
 *   SYM=0.5*(BigMat+BigMat');
 *   ASY=0.5*(BigMat-BigMat');
 */
void PlateEmbedded::computeSYMASY()
{
    int i, j, k;
    int idx, idxT;
    int h = bandSize;
    int hh = h+h;
    double val, valT;
   
    /* loop over rows */
    /* first the "complete" rows */
    idx = 12;
    for (i = 0; i < (ss - hh - 2); i++) {

	idx = (25*i)+12;

	/* diagonal */
	SYM->values[idx] = BigMat5x5->values[idx];
	ASY->values[idx] = 0.0;
	idx++;

	/* +1 */
	idxT = ((i+1)*25)+11;
	DO_SYM_ASY_ENTRY;

	/* +2 */
	idxT = ((i+2)*25)+10;
	DO_SYM_ASY_ENTRY;

	/* +h-2 */
	idxT = ((i+h-2)*25)+9;
	DO_SYM_ASY_ENTRY;

	/* +h-1 */
	idxT = ((i+h-1)*25)+8;
	DO_SYM_ASY_ENTRY;

	/* +h */
	idxT = ((i+h)*25)+7;
	DO_SYM_ASY_ENTRY;

	/* +h+1 */
	idxT = ((i+h+1)*25)+6;
	DO_SYM_ASY_ENTRY;

	/* +h+2 */
	idxT = ((i+h+2)*25)+5;
	DO_SYM_ASY_ENTRY;
	
	/* +2h-2 */
	idxT = ((i+hh-2)*25)+4;
	DO_SYM_ASY_ENTRY;

	/* +2h-1 */
	idxT = ((i+hh-1)*25)+3;
	DO_SYM_ASY_ENTRY;

	/* +2h */
	idxT = ((i+hh)*25)+2;
	DO_SYM_ASY_ENTRY;

	/* +2h+1 */
	idxT = ((i+hh+1)*25)+1;
	DO_SYM_ASY_ENTRY;

	/* +2h+2 */
	idxT = ((i+hh+2)*25)+0;
	DO_SYM_ASY_ENTRY;

	idx += 12;
    }

    /* and now the remainder */
    for (; i < ss; i++) {
	idx = (25*i)+12;
	/* diagonal */
	SYM->values[idx] = BigMat5x5->values[idx];
	ASY->values[idx] = 0.0;
	idx++;

	if (i < (ss-1)) {
	    /* +1 */
	    idxT = ((i+1)*25)+11;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-2)) {
	    /* +2 */
	    idxT = ((i+2)*25)+10;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-h+2)) {
	    /* +h-2 */
	    idxT = ((i+h-2)*25)+9;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-h+1)) {
	    /* +h-1 */
	    idxT = ((i+h-1)*25)+8;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-h)) {
	    /* +h */
	    idxT = ((i+h)*25)+7;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-h-1)) {
	    /* +h+1 */
	    idxT = ((i+h+1)*25)+6;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-h-2)) {
	    /* +h+2 */
	    idxT = ((i+h+2)*25)+5;
	    DO_SYM_ASY_ENTRY;
	}
	
	if (i < (ss-hh+2)) {
	    /* +2h-2 */
	    idxT = ((i+hh-2)*25)+4;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-hh+1)) {
	    /* +2h-1 */
	    idxT = ((i+hh-1)*25)+3;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-hh)) {
	    /* +2h */
	    idxT = ((i+hh)*25)+2;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-hh-1)) {
	    /* +2h+1 */
	    idxT = ((i+hh+1)*25)+1;
	    DO_SYM_ASY_ENTRY;
	}

	if (i < (ss-hh-2)) {
	    /* +2h+2 */
	    idxT = ((i+hh+2)*25)+0;
	    DO_SYM_ASY_ENTRY;
	}

	idx += 12;
    }

    if (isCircular) {
	j = 0;
	for (i = 0; i < ss; i++) {
	    if (TruePhi[j] == i) {
		j++;
	    }
	    else {
		for (k = 0; k < 25; k++) {
		    SYM->values[i*25+k] = 0.0;
		    ASY->values[i*25+k] = 0.0;
		}
	    }
	}
    }
}

/*
 * Implements:
 *   BigMat=DDDD_F+LOPW*LOPF;
 */
void PlateEmbedded::computeBigMat()
{
    int i, j, k;
    int H = bandSize;
    int HH = H+H;
    int N = BigMat->nrow;
    double diff;
    int offs;
    double val;

    double *v;

    int didx;

    //double lopw[9];
    double *lopw;

    double x = dlmmval;

    v = BigMat5x5->values;

    lopw = LOPW3x3->values;

    /* loop over all populated values in destination matrix */
    for (i = 0; i < N; i++) {
#define LOPF0 ((j+H+1) >= N || (j+H+1) < 0 ? 0.0 : LOPF3x3->values[((j+H+1)*9)])
#define LOPF1 ((j+H) >= N   || (j+H) < 0   ? 0.0 : LOPF3x3->values[((j+H)*9)+1])
#define LOPF2 ((j+H-1) >= N || (j+H-1) < 0 ? 0.0 : LOPF3x3->values[((j+H-1)*9)+2])

#define LOPF3 ((j+1) >= N   || (j+1) < 0   ? 0.0 : LOPF3x3->values[((j+1)*9)+3])
#define LOPF4 (j >= N       || j < 0       ? 0.0 : LOPF3x3->values[(j*9)+4])
#define LOPF5 ((j-1) >= N   || (j-1) < 0   ? 0.0 : LOPF3x3->values[((j-1)*9)+5])

#define LOPF6 ((j-H+1) >= N || (j-H+1) < 0 ? 0.0 : LOPF3x3->values[((j-H+1)*9)+6])
#define LOPF7 ((j-H) >= N   || (j-H) < 0   ? 0.0 : LOPF3x3->values[((j-H)*9)+7])
#define LOPF8 ((j-H-1) >= N || (j-H-1) < 0 ? 0.0 : LOPF3x3->values[((j-H-1)*9)+8])

	didx = DDDD_F->rowStart[i];
	k = BigMat->rowStart[i];
	/*
	 * If this is a fully populated row (as most of them are), unroll the inner loop
	 */
	if (((BigMat->rowStart[i+1] - k) == 25) && ((DDDD_F->rowStart[i+1] - didx) == 13)) {
	    j = BigMat->colIndex[k];
	    v[0   ] = lopw[0]*LOPF0;
	    j = BigMat->colIndex[k+1];
	    v[0+1 ] = lopw[0]*LOPF1 + lopw[1]*LOPF0;
	    j = BigMat->colIndex[k+2];
	    v[0+2 ] = lopw[0]*LOPF2 + lopw[1]*LOPF1 + lopw[2]*LOPF0
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+3];
	    v[0+3 ] = lopw[1]*LOPF2 + lopw[2]*LOPF1;
	    j = BigMat->colIndex[k+4];
	    v[0+4 ] = lopw[2]*LOPF2;

	    j = BigMat->colIndex[k+5];
	    v[0+5 ] = lopw[0]*LOPF3
		+ lopw[3]*LOPF0;
	    j = BigMat->colIndex[k+6];
	    v[0+6 ] = lopw[0]*LOPF4 + lopw[1]*LOPF3
		+ lopw[3]*LOPF1 + lopw[4]*LOPF0
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+7];
	    v[0+7 ] = lopw[0]*LOPF5 + lopw[1]*LOPF4 + lopw[2]*LOPF3
		+ lopw[3]*LOPF2 + lopw[4]*LOPF1 + lopw[5]*LOPF0
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+8];
	    v[0+8 ] = lopw[1]*LOPF5 + lopw[2]*LOPF4
		+ lopw[4]*LOPF2 + lopw[5]*LOPF1
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+9];
	    v[0+9 ] = lopw[2]*LOPF5
		+ lopw[5]*LOPF2;

	    j = BigMat->colIndex[k+10];
	    v[0+10] =  lopw[0]*LOPF6
		+ lopw[3]*LOPF3
		+ lopw[6]*LOPF0
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+11];
	    v[0+11] = lopw[0]*LOPF7 + lopw[1]*LOPF6
		+ lopw[3]*LOPF4 + lopw[4]*LOPF3
		+ lopw[6]*LOPF1 + lopw[7]*LOPF0
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+12];
	    v[0+12] = lopw[0]*LOPF8 + lopw[1]*LOPF7 + lopw[2]*LOPF6
		+ lopw[3]*LOPF5 + lopw[4]*LOPF4 + lopw[5]*LOPF3
		+ lopw[6]*LOPF2 + lopw[7]*LOPF1 + lopw[8]*LOPF0
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+13];
	    v[0+13] = lopw[1]*LOPF8 + lopw[2]*LOPF7
		+ lopw[4]*LOPF5 + lopw[5]*LOPF4
		+ lopw[7]*LOPF2 + lopw[8]*LOPF1
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+14];
	    v[0+14] = lopw[2]*LOPF8
		+ lopw[5]*LOPF5
		+ lopw[8]*LOPF2
		+ DDDD_F->values[didx++];

	    j = BigMat->colIndex[k+15];
	    v[0+15] = lopw[3]*LOPF6
		+ lopw[6]*LOPF3;
	    j = BigMat->colIndex[k+16];
	    v[0+16] = lopw[3]*LOPF7 + lopw[4]*LOPF6
		+ lopw[6]*LOPF4 + lopw[7]*LOPF3
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+17];
	    v[0+17] = lopw[3]*LOPF8 + lopw[4]*LOPF7 + lopw[5]*LOPF6
		+ lopw[6]*LOPF5 + lopw[7]*LOPF4 + lopw[8]*LOPF3
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+18];
	    v[0+18] = lopw[4]*LOPF8 + lopw[5]*LOPF7
		+ lopw[7]*LOPF5 + lopw[8]*LOPF4
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+19];
	    v[0+19] = lopw[5]*LOPF8
		+ lopw[8]*LOPF5;

	    j = BigMat->colIndex[k+20];
	    v[0+20] = lopw[6]*LOPF6;
	    j = BigMat->colIndex[k+21];
	    v[0+21] = lopw[6]*LOPF7 + lopw[7]*LOPF6;
	    j = BigMat->colIndex[k+22];
	    v[0+22] = lopw[6]*LOPF8 + lopw[7]*LOPF7 + lopw[8]*LOPF6
		+ DDDD_F->values[didx++];
	    j = BigMat->colIndex[k+23];
	    v[0+23] = lopw[7]*LOPF8 + lopw[8]*LOPF7;
	    j = BigMat->colIndex[k+24];
	    v[0+24] = lopw[8]*LOPF8;
	}
	else {
	    for (k = BigMat->rowStart[i]; k < BigMat->rowStart[i+1]; k++) {
		/* this is a dot product of row i of LOPW with col j of LOPF */
		j = BigMat->colIndex[k];
		
		offs = j - i;
		
		val = 0.0;
		/* add in the Laplacian */
		if (j == DDDD_F->colIndex[didx]) {
		    val = DDDD_F->values[didx++];
		}
		
		if (offs == -(HH+2)) {
		    val += lopw[0]*LOPF0;
		    v[0] = val;
		}
		else if (offs == -(HH+1)) {
		    val += lopw[0]*LOPF1 + lopw[1]*LOPF0;
		    v[1] = val;
		}
		else if (offs == -HH) {
		    val += lopw[0]*LOPF2 + lopw[1]*LOPF1 + lopw[2]*LOPF0;
		    v[2] = val;
		}
		else if (offs == -HH+1) {
		    val += lopw[1]*LOPF2 + lopw[2]*LOPF1;
		    v[3] = val;
		}
		else if (offs == -HH+2) {
		    val += lopw[2]*LOPF2;
		    v[4] = val;
		}
		
		else if (offs == -(H+2)) {
		    val += lopw[0]*LOPF3
			+ lopw[3]*LOPF0;
		    v[5] = val;
		}
		else if (offs == -(H+1)) {
		    val += lopw[0]*LOPF4 + lopw[1]*LOPF3
			+ lopw[3]*LOPF1 + lopw[4]*LOPF0;
		    v[6] = val;
		}
		else if (offs == -H) {
		    val += lopw[0]*LOPF5 + lopw[1]*LOPF4 + lopw[2]*LOPF3
			+ lopw[3]*LOPF2 + lopw[4]*LOPF1 + lopw[5]*LOPF0;
		    v[7] = val;
		}
		else if (offs == -H+1) {
		    val += lopw[1]*LOPF5 + lopw[2]*LOPF4
			+ lopw[4]*LOPF2 + lopw[5]*LOPF1;
		    v[8] = val;
		}
		else if (offs == -H+2) {
		    val += lopw[2]*LOPF5
			+ lopw[5]*LOPF2;
		    v[9] = val;
		}
		
		else if (offs == -2) {
		    val += lopw[0]*LOPF6
			+ lopw[3]*LOPF3
			+ lopw[6]*LOPF0;
		    v[10] = val;
		}
		else if (offs == -1) {
		    val += lopw[0]*LOPF7 + lopw[1]*LOPF6
			+ lopw[3]*LOPF4 + lopw[4]*LOPF3
			+ lopw[6]*LOPF1 + lopw[7]*LOPF0;
		    v[11] = val;
		}
		else if (offs == 0) {
		    val += lopw[0]*LOPF8 + lopw[1]*LOPF7 + lopw[2]*LOPF6
			+ lopw[3]*LOPF5 + lopw[4]*LOPF4 + lopw[5]*LOPF3
			+ lopw[6]*LOPF2 + lopw[7]*LOPF1 + lopw[8]*LOPF0;
		    v[12] = val;
		}
		else if (offs == 1) {
		    val += lopw[1]*LOPF8 + lopw[2]*LOPF7
			+ lopw[4]*LOPF5 + lopw[5]*LOPF4
			+ lopw[7]*LOPF2 + lopw[8]*LOPF1;
		    v[13] = val;
		}
		else if (offs == 2) {
		    val += lopw[2]*LOPF8
			+ lopw[5]*LOPF5
			+ lopw[8]*LOPF2;
		    v[14] = val;
		}
		
		else if (offs == H-2) {
		    val += lopw[3]*LOPF6
			+ lopw[6]*LOPF3;
		    v[15] = val;
		}
		else if (offs == H-1) {
		    val += lopw[3]*LOPF7 + lopw[4]*LOPF6
			+ lopw[6]*LOPF4 + lopw[7]*LOPF3;
		    v[16] = val;
		}
		else if (offs == H) {
		    val += lopw[3]*LOPF8 + lopw[4]*LOPF7 + lopw[5]*LOPF6
			+ lopw[6]*LOPF5 + lopw[7]*LOPF4 + lopw[8]*LOPF3;
		    v[17] = val;
		}
		else if (offs == H+1) {
		    val += lopw[4]*LOPF8 + lopw[5]*LOPF7
			+ lopw[7]*LOPF5 + lopw[8]*LOPF4;
		    v[18] = val;
		}
		else if (offs == H+2) {
		    val += lopw[5]*LOPF8
			+ lopw[8]*LOPF5;
		    v[19] = val;
		}
		
		else if (offs == HH-2) {
		    val += lopw[6]*LOPF6;
		    v[20] = val;
		}
		else if (offs == HH-1) {
		    val += lopw[6]*LOPF7 + lopw[7]*LOPF6;
		    v[21] = val;
		}
		else if (offs == HH) {
		    val += lopw[6]*LOPF8 + lopw[7]*LOPF7 + lopw[8]*LOPF6;
		    v[22] = val;
		}
		else if (offs == HH+1) {
		    val += lopw[7]*LOPF8 + lopw[8]*LOPF7;
		    v[23] = val;
		}
		else if (offs == HH+2) {
		    val += lopw[8]*LOPF8;
		    v[24] = val;
		}
	    }
	}
	v += 25;
	lopw += 9;
    }
}

/*
 * Implements:
 *   LOPW=(temp1*Dxxw+temp2*Dyyw-0.5*(temp3*DLMM+temp4*DLMP+temp5*DLPM+temp6*DLPP))*2*CF;
 *   LOPF=(temp1*Dxx_F+temp2*Dyy_F-0.5*(temp3*DLMM+temp4*DLMP+temp5+DLPM+temp6*DLPP))*BL;
 */
void PlateEmbedded::computeLOPWF()
{
    int i, j, k;
    double x = dlmmval;
    int H = bandSize;
    int offs;
    double val;
    
    /* loop over rows of LOPW */
    for (i = 0; i < LOPW->nrow; i++) {

	if ((LOPW->rowStart[i+1] - LOPW->rowStart[i]) == 9) {
	    /* fully populated row */
	    val = -0.5 * temp3[i] * x;
	    LOPW3x3->values[(i*9)+0] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+0] = BL * val;

	    val = (temp1[i] * x - 0.5 * ((temp3[i] * (-x)) + (temp4[i] * x)));
	    LOPW3x3->values[(i*9)+1] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+1] = BL * val;

	    val = -0.5 * temp4[i] * (-x);
	    LOPW3x3->values[(i*9)+2] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+2] = BL * val;

	    val = (temp2[i] * x - 0.5 * ((temp3[i] * (-x)) + (temp5[i] * x)));
	    LOPW3x3->values[(i*9)+3] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+3] = BL * val;

	    val = (temp1[i] * (-2.0*x) + temp2[i] * (-2.0*x) -
		   0.5 * ((temp3[i] * x) + (temp4[i] * (-x)) +
			  (temp5[i] * (-x)) + (temp6[i] * x)));
	    LOPW3x3->values[(i*9)+4] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+4] = BL * val;

	    val = (temp2[i] * x - 0.5 * ((temp4[i] * x) + (temp6[i] * (-x))));
	    LOPW3x3->values[(i*9)+5] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+5] = BL * val;

	    val = -0.5 * temp5[i] * (-x);
	    LOPW3x3->values[(i*9)+6] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+6] = BL * val;

	    val = (temp1[i] * x - 0.5 * ((temp5[i] * x) + (temp6[i] * (-x))));
	    LOPW3x3->values[(i*9)+7] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+7] = BL * val;

	    val = -0.5 * temp6[i] * x;
	    LOPW3x3->values[(i*9)+8] = 2.0 * CF * val;
	    LOPF3x3->values[(i*9)+8] = BL * val;	    

	}
	else {
	    /* loop over populated columns */
	    for (j = LOPW->rowStart[i]; j < LOPW->rowStart[i+1]; j++) {
		offs = LOPW->colIndex[j] - i;

		val = 0.0;
		if (offs < 0) {
		    if (offs == -(H+1)) {
			/* DLMM */
			val = -0.5 * temp3[i] * x;
			LOPW3x3->values[(i*9)+0] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+0] = BL * val;
		    }
		    else if (offs == -H) {
			/* DLMM, DLMP, Dxxw */
			val = (temp1[i] * x - 0.5 * ((temp3[i] * (-x)) + (temp4[i] * x)));
			LOPW3x3->values[(i*9)+1] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+1] = BL * val;
		    }
		    else if (offs == -H+1) {
			/* DLMP */
			val = -0.5 * temp4[i] * (-x);
			LOPW3x3->values[(i*9)+2] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+2] = BL * val;
		    }
		    else if (offs == -1) {
			/* DLMM, DLPM, Dyyw */
			val = (temp2[i] * x - 0.5 * ((temp3[i] * (-x)) + (temp5[i] * x)));
			LOPW3x3->values[(i*9)+3] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+3] = BL * val;
		    }
		}
		else if (offs == 0) {
		    /* everything! */
		    val = (temp1[i] * (-2.0*x) + temp2[i] * (-2.0*x) -
			   0.5 * ((temp3[i] * x) + (temp4[i] * (-x)) +
				  (temp5[i] * (-x)) + (temp6[i] * x)));
		    LOPW3x3->values[(i*9)+4] = 2.0 * CF * val;
		    LOPF3x3->values[(i*9)+4] = BL * val;
		}
		else { /* offs > 0 */
		    if (offs == 1) {
			/* DLMP, DLPP, Dyyw */
			val = (temp2[i] * x - 0.5 * ((temp4[i] * x) + (temp6[i] * (-x))));
			LOPW3x3->values[(i*9)+5] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+5] = BL * val;
		    }
		    else if (offs == H-1) {
			/* DLPM */
			val = -0.5 * temp5[i] * (-x);
			LOPW3x3->values[(i*9)+6] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+6] = BL * val;
		    }
		    else if (offs == H) {
			/* DLPM, DLPP, Dxxw */
			val = (temp1[i] * x - 0.5 * ((temp5[i] * x) + (temp6[i] * (-x))));
			LOPW3x3->values[(i*9)+7] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+7] = BL * val;
		    }
		    else if (offs == H+1) {
			/* DLPP */
			val = -0.5 * temp6[i] * x;
			LOPW3x3->values[(i*9)+8] = 2.0 * CF * val;
			LOPF3x3->values[(i*9)+8] = BL * val;
		    }
		}
	    }
	}
    }

    if (isCircular) {
	j = 0;
	for (i = 0; i < ss; i++) {
	    if (TruePhi[j] == i) {
		j++;
	    }
	    else {
		for (k = 0; k < 9; k++) {
		    LOPW3x3->values[i*9+k] = 0.0;
		    LOPF3x3->values[i*9+k] = 0.0;
		}
	    }
	}
    }
}

/*
 * Implements:
 *   temp1=Dyyw*w1;
 *   temp2=Dxxw*w1;
 *   temp3=DLMM*w1;
 *   temp4=DLMP*w1;
 *   temp5=DLPM*w1;
 *   temp6=DLPP*w1;
 *
 * The original Matlab version stores tempX as diagonal matrices, we just use
 * vectors.
 */
void PlateEmbedded::computeTemp1_6()
{
    int i;
    double val;
    int h = bandSize;
    int N = ss;
    double *v1, *v2, *v3, *v4, *v5, *v6;
    double w0v, w1v, w2, w3, w4, w5, w6, w7, w8;

    v1 = &Dyyw3->values[0];
    v2 = &Dxxw3->values[0];
    v3 = &DLMM3->values[0];
    v4 = &DLMP3->values[0];
    v5 = &DLPM3->values[0];
    v6 = &DLPP3->values[0];
    for (i = 0; i < (h+1); i++) {
	if (i >= (h+1)) w0v = u1[i-h-1];
	else w0v = 0.0;

	if (i >= (h)) w1v = u1[i-h];
	else w1v = 0.0;

	if (i >= (h-1)) w2 = u1[i-h+1];
	else w2 = 0.0;

	if (i >= 1) w3 = u1[i-1];
	else w3 = 0.0;

	w4 = u1[i];
	w5 = u1[i+1];
	w6 = u1[i+h-1];
	w7 = u1[i+h];
	w8 = u1[i+h+1];

	temp1[i] = w3*v1[3] + w4*v1[4] + w5*v1[5];
	temp2[i] = w1v*v2[1] + w4*v2[4] + w7*v2[7];
	temp3[i] = w0v*v3[0] + w1v*v3[1] + 
	    w3*v3[3] + w4*v3[4];
	temp4[i] = w1v*v4[1] + w2*v4[2] +
	    w4*v4[4] + w5*v4[5];
	temp5[i] = w3*v5[3] + w4*v5[4] +
	    w6*v5[6] + w7*v5[7];
	temp6[i] = w4*v6[4] + w5*v6[5] +
	    w7*v6[7] + w8*v6[8];

	v1 += 9;
	v2 += 9;
	v3 += 9;
	v4 += 9;
	v5 += 9;
	v6 += 9;
    }
    for (; i < (N-h-1); i++) {
	w0v = u1[i-h-1];
	w1v = u1[i-h];
	w2 = u1[i-h+1];
	w3 = u1[i-1];
	w4 = u1[i];
	w5 = u1[i+1];
	w6 = u1[i+h-1];
	w7 = u1[i+h];
	w8 = u1[i+h+1];

	temp1[i] = w3*v1[3] + w4*v1[4] + w5*v1[5];
	temp2[i] = w1v*v2[1] + w4*v2[4] + w7*v2[7];
	temp3[i] = w0v*v3[0] + w1v*v3[1] + 
	    w3*v3[3] + w4*v3[4];
	temp4[i] = w1v*v4[1] + w2*v4[2] +
	    w4*v4[4] + w5*v4[5];
	temp5[i] = w3*v5[3] + w4*v5[4] +
	    w6*v5[6] + w7*v5[7];
	temp6[i] = w4*v6[4] + w5*v6[5] +
	    w7*v6[7] + w8*v6[8];

	v1 += 9;
	v2 += 9;
	v3 += 9;
	v4 += 9;
	v5 += 9;
	v6 += 9;
    }
    for (; i < N; i++) {
	w0v = u1[i-h-1];
	w1v = u1[i-h];
	w2 = u1[i-h+1];
	w3 = u1[i-1];
	w4 = u1[i];

	if (i < (N-1)) w5 = u1[i+1];
	else w5 = 0.0;
	
	if (i < (N-h+1)) w6 = u1[i+h-1];
	else w6 = 0.0;

	if (i < (N-h)) w7 = u1[i+h];
	else w7 = 0.0;

	if (i < (N-h-1)) w8 = u1[i+h+1];
	else w8 = 0.0;

	temp1[i] = w3*v1[3] + w4*v1[4] + w5*v1[5];
	temp2[i] = w1v*v2[1] + w4*v2[4] + w7*v2[7];
	temp3[i] = w0v*v3[0] + w1v*v3[1] + 
	    w3*v3[3] + w4*v3[4];
	temp4[i] = w1v*v4[1] + w2*v4[2] +
	    w4*v4[4] + w5*v4[5];
	temp5[i] = w3*v5[3] + w4*v5[4] +
	    w6*v5[6] + w7*v5[7];
	temp6[i] = w4*v6[4] + w5*v6[5] +
	    w7*v6[7] + w8*v6[8];

	v1 += 9;
	v2 += 9;
	v3 += 9;
	v4 += 9;
	v5 += 9;
	v6 += 9;
    }    
}

void PlateEmbedded::applyIMat(double *source, double *dest, double scalar)
{
    // FIXME: add optimised version eventually
    int i;
    CSR_matrix_vector_mult(IMat, source, dest);
    for (i = 0; i < ss; i++) {
	dest[i] *= scalar;
    }
}

/*==============================================================================
 *
 * Initialisation functions
 *
 *============================================================================*/
void PlateEmbedded::setInterpolation(CSRmatrix *IMat, CSRmatrix *JMat, double *buffer,
				     double cplfac)
{
    CSRmatrix *cplfacIMat, *cplfacIMatJMat;

    this->IMat = IMat;
    this->JMat = JMat;
    interpBuffer = buffer;
    this->cplfac = cplfac;

    // see if we can make a banded IMat*JMat
    cplfacIMat = CSR_duplicate(IMat);
    CSR_scalar_mult(cplfacIMat, cplfac);

    cplfacIMatJMat = CSR_matrix_multiply(cplfacIMat, JMat);
    
    cplfacIMatJMat3x3 = allocate3x3(ss, bandSize);
    if (!csrTo3x3(cplfacIMatJMat, cplfacIMatJMat3x3)) {
	/* doesn't fit the 3x3 banded structure */
	free3x3(cplfacIMatJMat3x3);
	cplfacIMatJMat3x3 = NULL;

	logMessage(1, "Using 2-stage interpolation");
    }

    CSR_free(cplfacIMat);
    CSR_free(cplfacIMatJMat);

    Diff_tmp1 = new double[JMat->nrow];
}


void PlateEmbedded::init(double nu, double rho, double E, double thickness,
			 double tension, double lx, double ly, double t60_0, double sig1,
			 double cx, double cy, double cz,
			 bool isMembrane, bool isCircular)
{
    double K, sig0, c;

    CSRmatrix *DDDD, *temp, *tempm2;

    CSRmatrix *Dxxw, *Dyyw, *Dxx_F = NULL, *Dyy_F = NULL;
    CSRmatrix *DLMM, *DLMP, *DLPM, *DLPP;

    double gridx, gridy;

    int i, j, jdx, tpidx;

    // total number of grid points during setup. The final number of grid points actually
    // used will be in (lower case) instance variables nx and ny
    int Nx, Ny;

    // state sizes during setup. Final state size will be in (lower case) instance
    // variable ss
    int SS, SSPhi;

    SettingsManager *sm = SettingsManager::getInstance();

    logMessage(1, "Setting up plate: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
	       nu, rho, E, thickness, tension, lx, ly, t60_0, sig1, cx, cy, cz);

    profiler = new Profiler(14, profileNames);

    // save values
    this->lx = lx;
    this->ly = ly;
    this->cx = cx;
    this->cy = cy;
    this->cz = cz;
    this->rho = rho;
    this->H = thickness;
    this->E = E;
    this->nu = nu;
    this->T60 = t60_0;
    this->sig1 = sig1;
    this->isMembrane = isMembrane;
    this->isCircular = isCircular;
    this->T = tension;

    IMat = NULL;
    JMat = NULL;
    interpBuffer = NULL;
    cplfacIMatJMat3x3 = NULL;
    Diff_tmp1 = NULL;

    linear = sm->getBoolSetting(name, "linear");
    symmetric = sm->getBoolSetting(name, "symmetric");
    iterinv = sm->getIntSetting(name, "iterinv");

    if (iterinv < 0) {
	// different defaults for plates and membranes
	if (!isMembrane) iterinv = 5;
	else iterinv = 8;
    }

    GlobalSettings *gs = GlobalSettings::getInstance();
    double SR = gs->getSampleRate();

    energy = NULL;
    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps()];
    }
    
    // compute scalars first
    D = (E * H * H * H) / (12.0 * (1.0 - nu*nu));
    K = sqrt(D / (rho * H));
    sig0 = (6.0 * log(10.0)) / T60;
    k = 1.0 / SR;

    // sort out loss
    switch (sm->getIntSetting(name, "loss_mode")) {
    case 0:
	sig0 = 0.0;
	this->sig1 = 0.0;
	break;
    case -1:
	sig0 = 0.0;
	break;
    }

    if (!isMembrane) {
	h = sqrt(sqrt(16.0*K*K*k*k) + 8.0*sig1*k);
    }
    else {
	c = sqrt(T / (rho * H));
	h = sqrt(c*c*k*k + 4.0*sig1*k +
		 sqrt(((c*c*k*k + 4.0*sig1*k) * (c*c*k*k + 4.0*sig1*k)) + 16.0*K*K*k*k));
    }
    h = h * sm->getDoubleSetting(name, "fixpar");

    // for circular plates/membranes, Lx gives radius
    if (isCircular) {
	lx = 2.0 * lx;
	ly = lx;
	this->lx = lx;
	this->ly = ly;
    }

    // determine number of grid points and grid spacing
    Nx = (int)floor(lx / h);
    Ny = (int)floor(ly / h);
    h = lx / (double)Nx; // recompute h

    SS = (Nx+1) * (Ny+1);
    logMessage(1, "D=%f, K=%f, sig0=%f, h=%f, SS=%d", D, K, sig0, h, SS);

    a0 = (sig0 * k) / (rho * H);
    alpha = (k*k) / ((1.0 + a0) * h * h * rho * H);
    bowFactor = k / (2.0 * (1.0 + k * sig0) * h * h);
    BL = (0.5 * k * k) / (1.0 + a0) / (rho * H);
    CF = 0.5 * E * H;
    logMessage(1, "Scalars: %.20f, %.20f, %.20f\n", alpha, BL, CF);

    // generate index set TruePhi and boolean mask true_Psi
    if (!isCircular) {
	SSPhi = (Nx-1)*(Ny-1);
	TruePhiSize = SSPhi;
	bandSize = Ny - 1;

	allocateState(Nx-1, Ny-1);
	
	/* generate index set TruePhi */
	TruePhi = new int[TruePhiSize];
	jdx = 0;
	tpidx = 0;
	for (i = 0; i < (Nx+1); i++) {
	    for (j = 0; j < (Ny+1); j++) {
		if ((i > 0) && (i < Nx) && (j > 0) && (j < Ny)) {
		    TruePhi[tpidx] = jdx;
		    tpidx++;
		}
		jdx++;
	    }
	}
    }
    else {
	double radsq = ((lx/2.0) * (lx/2.0));

	SSPhi = SS;
	bandSize = Ny + 1;

	allocateState(Nx+1, Ny+1);

	/* generate circular mask */
	// first count points within mask
	TruePhiSize = 0;
	gridx = -((double)Nx / 2.0) * h;
	for (i = 0; i < (Nx+1); i++) {
	    gridy = -((double)Ny / 2.0) * h;
	    for (j = 0; j < (Ny+1); j++) {
		if (((gridx*gridx)+(gridy*gridy)) < radsq) {
		    TruePhiSize++;
		}
		gridy += h;
	    }
	    gridx += h;
	}

	// now generate TruePhi
	TruePhi = new int[TruePhiSize];
	jdx = 0;
	tpidx = 0;
	gridx = -((double)Nx / 2.0) * h;
	for (i = 0; i < (Nx+1); i++) {
	    gridy = -((double)Ny / 2.0) * h;
	    for (j = 0; j < (Ny+1); j++) {
		if (((gridx*gridx)+(gridy*gridy)) < radsq) {
		    TruePhi[tpidx] = jdx;
		    tpidx++;
		}
		jdx++;
		gridy += h;
	    }
	    gridx += h;
	}
    }
    logMessage(1, "SSPhi: %d, TruePhiSize: %d", SSPhi, TruePhiSize);

    /* generate biharmonic */ 
    logMessage(1, "Generating biharmonic");
    myBiharmRect(nu, Nx+1, Ny+1, 0, 1, &DDDD, &DD, &Dxxw, &Dyyw, &temp);
    CSR_free(temp);
    Bw = CSR_duplicate(DD);

    if (!linear) {
	logMessage(1, "Generating DDDD_F");
	if (!isCircular) {
	    myAiryRect(nu, Nx+1, Ny+1, 0, 1, &DDDD_F, &DD_F, &Dxx_F, &Dyy_F);
	}
	else {
	    newAiryCircular(Nx+1, &DDDD_F, &DD_F);
	    
	    Dxx_F = CSR_duplicate(Dxxw);
	    Dyy_F = CSR_duplicate(Dyyw);
	}
    }

    /* crop matrices to correct area */
    logMessage(1, "Cropping matrices");
    if (!isCircular) {
	CSR_cut_cols(Dxxw, TruePhi, TruePhiSize);
	CSR_cut_rows(Dxxw, TruePhi, TruePhiSize);
	CSR_cut_cols(Dyyw, TruePhi, TruePhiSize);
	CSR_cut_rows(Dyyw, TruePhi, TruePhiSize);

	if (!linear) {
	    CSR_cut_cols(Dxx_F, TruePhi, TruePhiSize);
	    CSR_cut_rows(Dxx_F, TruePhi, TruePhiSize);
	    CSR_cut_cols(Dyy_F, TruePhi, TruePhiSize);
	    CSR_cut_rows(Dyy_F, TruePhi, TruePhiSize);
	    
	    CSR_cut_cols(DDDD_F, TruePhi, TruePhiSize);
	    CSR_cut_rows(DDDD_F, TruePhi, TruePhiSize);

	    CSR_cut_cols(DD_F, TruePhi, TruePhiSize);
	}

	CSR_cut_cols(DD, TruePhi, TruePhiSize);
	CSR_cut_cols(DDDD, TruePhi, TruePhiSize);
	CSR_cut_rows(DDDD, TruePhi, TruePhiSize);
	CSR_cut_cols(Bw, TruePhi, TruePhiSize);
	CSR_cut_rows(Bw, TruePhi, TruePhiSize);
    }
    else {
	CSR_zero_cols(Dxxw, TruePhi, TruePhiSize);
	CSR_zero_rows(Dxxw, TruePhi, TruePhiSize);
	CSR_zero_cols(Dyyw, TruePhi, TruePhiSize);
	CSR_zero_rows(Dyyw, TruePhi, TruePhiSize);
	
	if (!linear) {
	    CSR_zero_cols(Dxx_F, TruePhi, TruePhiSize);
	    CSR_zero_rows(Dxx_F, TruePhi, TruePhiSize);
	    CSR_zero_cols(Dyy_F, TruePhi, TruePhiSize);
	    CSR_zero_rows(Dyy_F, TruePhi, TruePhiSize);
	
	    CSR_zero_cols(DDDD_F, TruePhi, TruePhiSize);
	    CSR_zero_rows(DDDD_F, TruePhi, TruePhiSize);

	    CSR_zero_cols(DD_F, TruePhi, TruePhiSize);
	}

	CSR_zero_cols(DD, TruePhi, TruePhiSize);
	CSR_zero_cols(DDDD, TruePhi, TruePhiSize);
	CSR_zero_rows(DDDD, TruePhi, TruePhiSize);
	CSR_zero_cols(Bw, TruePhi, TruePhiSize);
	CSR_zero_rows(Bw, TruePhi, TruePhiSize);
    }

    // generate the DL matrices
    generateDLMatrices(&DLMM, &DLMP, &DLPM, &DLPP, Nx, Ny);

    // generate difference matrices B, C, Bw
    generateDifferenceMatrices(DDDD, Dxxw, Dyyw, K, a0, k, c);
    CSR_free(DDDD);

    // state is allocated by call to allocateState in Component2D above
    // allocate and initialise other vectors here
    if (!linear) {
	Phi = new double[SSPhi];
	Phi1 = new double[SSPhi];
	Phi0 = new double[SSPhi];

	memset(Phi, 0, SSPhi * sizeof(double));
	memset(Phi1, 0, SSPhi * sizeof(double));
	memset(Phi0, 0, SSPhi * sizeof(double));

	temp1 = new double[SSPhi];
	temp2 = new double[SSPhi];
	temp3 = new double[SSPhi];
	temp4 = new double[SSPhi];
	temp5 = new double[SSPhi];
	temp6 = new double[SSPhi];

	aa = new double[SSPhi];
	bb = new double[SSPhi];
	BigRight = new double[SSPhi];
    }

    Right = new double[SSPhi];
    vtemp = new double[SSPhi];

    // create PCG if necessary
    if (!linear) {
	logMessage(1, "Creating PCG solver");
#ifdef USE_SSE
	pcg = pcgCreateSSE(DDDD_F, sm->getDoubleSetting(name, "pcg_tolerance"),
	    sm->getIntSetting(name, "pcg_max_it"));
#else
	pcg = pcgCreateCholesky(DDDD_F, sm->getDoubleSetting(name, "pcg_tolerance"),
	    sm->getIntSetting(name, "pcg_max_it"));
#endif

	// pre-allocate matrix structures
	logMessage(1, "Pre-allocating matrix structures");
	preAllocateMatrices(DLMM, DLMP, DLPM, DLPP, Dxxw, Dyyw, Dxx_F, Dyy_F);

	// convert matrices to banded
	logMessage(1, "Converting to banded matrices");
	BigMat5x5 = allocate5x5(SSPhi, bandSize);

	SYM = allocate5x5(SSPhi, bandSize);
	ASY = allocate5x5(SSPhi, bandSize);

	LOPW3x3 = allocate3x3(SSPhi, bandSize);
	LOPF3x3 = allocate3x3(SSPhi, bandSize);

	DLMM3 = allocate3x3(SSPhi, bandSize);
	DLMP3 = allocate3x3(SSPhi, bandSize);
	DLPM3 = allocate3x3(SSPhi, bandSize);
	DLPP3 = allocate3x3(SSPhi, bandSize);
	Dxxw3 = allocate3x3(SSPhi, bandSize);
	Dyyw3 = allocate3x3(SSPhi, bandSize);
	Dxx_F3 = allocate3x3(SSPhi, bandSize);
	Dyy_F3 = allocate3x3(SSPhi, bandSize);
	
	if (!csrTo3x3(DLMM, DLMM3)) {
	    logMessage(5, "Internal error: DLMM doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(DLMP, DLMP3)) {
	    logMessage(5, "Internal error: DLMP doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(DLPM, DLPM3)) {
	    logMessage(5, "Internal error: DLPM doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(DLPP, DLPP3)) {
	    logMessage(5, "Internal error: DLPP doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(Dxxw, Dxxw3)) {
	    logMessage(5, "Internal error: Dxxw doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(Dyyw, Dyyw3)) {
	    logMessage(5, "Internal error: Dyyw doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(Dxx_F, Dxx_F3)) {
	    logMessage(5, "Internal error: Dxx_F doesn't fit banded structure!!\n");
	    exit(1);
	}
	if (!csrTo3x3(Dyy_F, Dyy_F3)) {
	    logMessage(5, "Internal error: Dyy_F doesn't fit banded structure!!\n");
	    exit(1);
	}
	
	// compute dlmmval
	dlmmval = (1.0 / h) * (1.0 / h);
	logMessage(1, "dlmmval=%.20f\n", dlmmval);

	// free temporary matrices
	CSR_free(DLMM);
	CSR_free(DLMP);
	CSR_free(DLPM);
	CSR_free(DLPP);
	CSR_free(Dxx_F);
	CSR_free(Dyy_F);
    }

    CSR_free(Dxxw);
    CSR_free(Dyyw);

    if (energy) {
	energytmp = new double[SS];
	energytmp2 = new double[SS];

	if (!isCircular) {
	    CSR_cut_cols(Dxp, TruePhi, TruePhiSize);
	    CSR_cut_cols(Dyp, TruePhi, TruePhiSize);
	}
	else {
	    CSR_zero_cols(Dxp, TruePhi, TruePhiSize);
	    CSR_zero_cols(Dyp, TruePhi, TruePhiSize);
	}
    }
    else {
	CSR_free(DD);
	if (!linear) {
	    CSR_free(DD_F);
	}
	CSR_free(Dxp);
	CSR_free(Dyp);
    }
}


void PlateEmbedded::generateDLMatrices(CSRmatrix **DLMM, CSRmatrix **DLMP, CSRmatrix **DLPM,
				       CSRmatrix **DLPP, int Nx, int Ny)
{
    double *diag1, *diag2;
    CSRmatrix *temp, *Dxp, *Dyp;

    matrix_3x3_t *Dxp3, *Dyp3, *Dxm3, *Dym3;
    matrix_5x5_t *DLMM5, *DLMP5, *DLPM5, *DLPP5;

    int SS = (Nx+1) * (Ny+1);

    logMessage(1, "Generating DL matrices");
    /* first need Dxp and Dxm */
    diag1 = new double[Nx+1];
    memset(diag1, 0, (Nx+1)*sizeof(double));
    diag1[0] = -1.0;
    diag1[1] = 1.0;
    diag2 = new double[Nx+1];
    memset(diag2, 0, (Nx+1)*sizeof(double));
    diag2[0] = -1.0;

    temp = CSR_toeplitz(diag2, Nx+1, diag1, Nx+1);
    Dxp = CSR_kron_mat_eye(temp, Ny+1);
    CSR_scalar_mult(Dxp, 1.0 / h);
    CSR_free(temp);
    delete[] diag1;
    delete[] diag2;

    /* Dyp and Dym */
    diag1 = new double[Ny+1];
    memset(diag1, 0, (Ny+1)*sizeof(double));
    diag1[0] = -1.0;
    diag1[1] = 1.0;
    diag2 = new double[Ny+1];
    memset(diag2, 0, (Ny+1)*sizeof(double));
    diag2[0] = -1.0;

    temp = CSR_toeplitz(diag2, Ny+1, diag1, Ny+1);
    Dyp = CSR_kron_eye_mat(temp, Nx+1);
    CSR_scalar_mult(Dyp, 1.0 / h);
    CSR_free(temp);
    delete[] diag1;
    delete[] diag2;

    // need these for energy check
    this->Dxp = Dxp;
    this->Dyp = Dyp;

    // in linear case, only need these
    if (linear) return;

    Dxp3 = allocate3x3(SS, Ny+1);
    Dxm3 = allocate3x3(SS, Ny+1);
    csrTo3x3(Dxp, Dxp3);
    m3x3_transpose(Dxp3, Dxm3);
    m3x3_scalar_mult(Dxm3, -1.0);

    Dyp3 = allocate3x3(SS, Ny+1);
    Dym3 = allocate3x3(SS, Ny+1);
    csrTo3x3(Dyp, Dyp3);
    m3x3_transpose(Dyp3, Dym3);
    m3x3_scalar_mult(Dym3, -1.0);


    /* now generate final DL matrices */
    DLMM5 = allocate5x5(SS, Ny+1);
    DLMP5 = allocate5x5(SS, Ny+1);
    DLPM5 = allocate5x5(SS, Ny+1);
    DLPP5 = allocate5x5(SS, Ny+1);

    m3x3_matrix_multiply(Dxm3, Dym3, DLMM5);
    m3x3_matrix_multiply(Dxm3, Dyp3, DLMP5);
    m3x3_matrix_multiply(Dxp3, Dym3, DLPM5);
    m3x3_matrix_multiply(Dxp3, Dyp3, DLPP5);

    *DLMM = m5x5ToCSR(DLMM5);
    *DLMP = m5x5ToCSR(DLMP5);
    *DLPM = m5x5ToCSR(DLPM5);
    *DLPP = m5x5ToCSR(DLPP5);

    free3x3(Dxm3);
    free3x3(Dym3);
    free3x3(Dxp3);
    free3x3(Dyp3);

    free5x5(DLMM5);
    free5x5(DLMP5);
    free5x5(DLPM5);
    free5x5(DLPP5);

    /* crop to TruePhi */
    if (!isCircular) {
	CSR_cut_cols(*DLMM, TruePhi, TruePhiSize);
	CSR_cut_rows(*DLMM, TruePhi, TruePhiSize);
	CSR_cut_cols(*DLMP, TruePhi, TruePhiSize);
	CSR_cut_rows(*DLMP, TruePhi, TruePhiSize);
	CSR_cut_cols(*DLPM, TruePhi, TruePhiSize);
	CSR_cut_rows(*DLPM, TruePhi, TruePhiSize);
	CSR_cut_cols(*DLPP, TruePhi, TruePhiSize);
	CSR_cut_rows(*DLPP, TruePhi, TruePhiSize);
    }
    else {
	CSR_zero_cols(*DLMM, TruePhi, TruePhiSize);
	CSR_zero_rows(*DLMM, TruePhi, TruePhiSize);
	CSR_zero_cols(*DLMP, TruePhi, TruePhiSize);
	CSR_zero_rows(*DLMP, TruePhi, TruePhiSize);
	CSR_zero_cols(*DLPM, TruePhi, TruePhiSize);
	CSR_zero_rows(*DLPM, TruePhi, TruePhiSize);
	CSR_zero_cols(*DLPP, TruePhi, TruePhiSize);
	CSR_zero_rows(*DLPP, TruePhi, TruePhiSize);
    }
}

// modifies DDDD!
void PlateEmbedded::generateDifferenceMatrices(CSRmatrix *DDDD, CSRmatrix *Dxxw,
					       CSRmatrix *Dyyw,
					       double K, double a0,
					       double k, double c)
{
    CSRmatrix *temp, *tempm2;

    logMessage(1, "Generating difference matrices");
    if (!isMembrane) {
	temp = CSR_create_eye(ss);
	CSR_scalar_mult(temp, 2.0);
	CSR_scalar_mult(DDDD, -K*K*k*k);
	B = CSR_matrix_add(temp, DDDD);
	CSR_scalar_mult(B, 1.0/(1.0+a0));
	CSR_free(temp);
	
	CSR_scalar_mult(Bw, (2.0*sig1*k)/(1.0+a0)/(rho*H));
    }
    else {
	// get rid of original DDDD and DD, they don't work here
	CSR_free(Bw);
	CSR_free(DD);

	// compute correct versions
	DD = CSR_matrix_add(Dxxw, Dyyw);
	CSRmatrix *DDDD2 = CSR_matrix_multiply(DD, DD);
	Bw = CSR_duplicate(DD);

	// compute B
	temp = CSR_create_eye(ss);
	CSR_scalar_mult(temp, 2.0);
	CSR_scalar_mult(DDDD2, -K*K*k*k);
	tempm2 = CSR_matrix_add(temp, DDDD2);
	CSR_free(temp);
	temp = CSR_duplicate(Bw); // get DD
	CSR_scalar_mult(temp, c*c*k*k);
	B = CSR_matrix_add(temp, tempm2);
	CSR_scalar_mult(B, 1.0/(1.0+a0));
	CSR_free(temp);
	CSR_free(tempm2);
	CSR_free(DDDD2);	

	// compute Bw
	CSR_scalar_mult(Bw, (2.0*sig1*k)/(1.0+a0));
    }

    // C is the same for both cases
    C = CSR_create_eye(ss);
    CSR_scalar_mult(C, (1.0-a0)/(1.0+a0));

    /* factor Bw into B and C now */
    temp = CSR_matrix_add(B, Bw);
    CSR_free(B);
    B = temp;

    temp = CSR_matrix_add(C, Bw);
    CSR_free(C);
    C = temp;
}

int PlateEmbedded::preAllocateMatrices(CSRmatrix *DLMM, CSRmatrix *DLMP, CSRmatrix *DLPM,
				       CSRmatrix *DLPP, CSRmatrix *Dxxw, CSRmatrix *Dyyw,
				       CSRmatrix *Dxx_F, CSRmatrix *Dyy_F)
{
    CSRmatrix *tmpm1, *tmpm2, *tmpm3, *tmpm4, *tmpm5, *tmpm6, *tmpm7;

    /* intermediate values */
    tmpm1 = CSR_matrix_add(DLMM, DLMP);
    tmpm2 = CSR_matrix_add(DLPM, DLPP);
    tmpm3 = CSR_matrix_add(tmpm1, tmpm2);

    tmpm4 = CSR_matrix_add(Dxxw, Dyyw);
    tmpm5 = CSR_duplicate(tmpm3);
    tmpm6 = CSR_matrix_add(Dxx_F, Dyy_F);

    /* LOPW and LOPF */
    LOPW = CSR_matrix_add(tmpm4, tmpm5);
    LOPF = CSR_matrix_add(tmpm6, tmpm3);

    /* tmpm7 and BigMat */
    tmpm7 = CSR_matrix_multiply(LOPW, LOPF);
    BigMat = CSR_matrix_add(DDDD_F, tmpm7);

    CSR_free(tmpm1);
    CSR_free(tmpm2);
    CSR_free(tmpm3);
    CSR_free(tmpm4);
    CSR_free(tmpm5);
    CSR_free(tmpm6);
    CSR_free(tmpm7);
    return 1;
}

int PlateEmbedded::newAiryCircular(int S, CSRmatrix **DDDD_F, CSRmatrix **DD_F)
{
    CSRmatrix *DxF, *Dxx_1F, *DyF, *Dyy_1F;
    double *diag1, *diag2;
    CSRmatrix *Dxy_1F, *Dxy_2F;
    CSRmatrix *Dxx_2F, *Dyy_2F;
    int l, m;
    int I, J;
    CSRmatrix *temp1, *temp2, *temp3, *temp4;

    matrix_5x5_t *DDDD_F5;
    matrix_3x3_t *Dxx_1F3, *Dxx_2F3, *Dyy_1F3, *Dyy_2F3;
    matrix_3x3_t *DD_F3;
    matrix_5x5_t *temp51, *temp52, *temp53, *temp54;

    /* generate Dxx and Dyy */
    diag1 = new double[S];
    memset(diag1, 0, S * sizeof(double));
    diag1[0] = -2.0;
    diag1[1] = 1.0;
    
    DxF = CSR_sym_toeplitz(diag1, S);
    Dxx_1F = CSR_kron_mat_eye(DxF, S);
    Dxx_2F = CSR_duplicate(Dxx_1F);

    diag2 = new double[S];
    memset(diag2, 0, S * sizeof(double));
    diag2[0] = -2.0;
    diag2[1] = 1.0;
    
    DyF = CSR_sym_toeplitz(diag2, S);
    Dyy_1F = CSR_kron_eye_mat(DyF, S);
    Dyy_2F = CSR_duplicate(Dyy_1F);

    delete[] diag1;
    delete[] diag2;
    CSR_free(DxF);
    CSR_free(DyF);

    /* generate Dxy */
    Dxy_2F = (CSRmatrix*)malloc(sizeof(CSRmatrix));
    CSR_setup(Dxy_2F, ((S-1)*(S-1)), S*S, ((S-1)*(S-1)*4));
    
    for (l = 0; l < (S-1); l++) {
	for (m = 0; m < (S-1); m++) {
	    I = m + (l * (S-1));
	    J = I + l;
	    CSRSetValue(Dxy_2F, I, J, 1.0);
	    CSRSetValue(Dxy_2F, I, J+1, -1.0);
	    CSRSetValue(Dxy_2F, I, J+S, -1.0);
	    CSRSetValue(Dxy_2F, I, J+S+1, 1.0);
	}
    }

    Dxy_1F = CSR_transpose(Dxy_2F);

    /* boundary conditions */
    for (l = 0; l < S; l++) {
	I = l;
	CSR_zero_row(Dxx_2F, I);
	CSRSetValue(Dxx_2F, I, I, -1.0);
	CSRSetValue(Dxx_2F, I, I+S, 1.0);
	I = l + (S-1)*S;
	CSR_zero_row(Dxx_2F, I);
	CSRSetValue(Dxx_2F, I, I, -1.0);
	CSRSetValue(Dxx_2F, I, I-S, 1.0);

	I = l * S;
	CSR_zero_row(Dyy_2F, I);
	CSRSetValue(Dyy_2F, I, I, -1.0);
	CSRSetValue(Dyy_2F, I, I+1, 1.0);
	I = (l * S) + (S - 1);
	CSR_zero_row(Dyy_2F, I);
	CSRSetValue(Dyy_2F, I, I, -1.0);
	CSRSetValue(Dyy_2F, I, I-1, 1.0);
    }

    /* remove rows not in circular mask */
    I = 0;
    for (l = 0; l < ss; l++) {
	if (TruePhi[I] == l) {
	    I++;
	}
	else {
	    CSR_zero_row(Dxx_1F, l);
	    CSR_zero_row(Dyy_1F, l);
	    CSR_zero_row(Dxy_1F, l);
	}
    }
    
    CSR_scalar_mult(Dxx_1F, (1.0 / (h*h)));
    CSR_scalar_mult(Dyy_1F, (1.0 / (h*h)));
    CSR_scalar_mult(Dxy_1F, (1.0 / (h*h)));
    CSR_scalar_mult(Dxx_2F, (1.0 / (h*h)));
    CSR_scalar_mult(Dyy_2F, (1.0 / (h*h)));
    CSR_scalar_mult(Dxy_2F, (1.0 / (h*h)));    

    /* convert matrices to 3x3 form */
    Dxx_1F3 = allocate3x3(S*S, S);
    Dyy_1F3 = allocate3x3(S*S, S);
    Dxx_2F3 = allocate3x3(S*S, S);
    Dyy_2F3 = allocate3x3(S*S, S);
    csrTo3x3(Dxx_1F, Dxx_1F3);
    csrTo3x3(Dyy_1F, Dyy_1F3);
    csrTo3x3(Dxx_2F, Dxx_2F3);
    csrTo3x3(Dyy_2F, Dyy_2F3);    

    /* compute DD_F */
    DD_F3 = allocate3x3(S*S, S);
    m3x3_matrix_add(Dxx_2F3, Dyy_2F3, DD_F3);
    *DD_F = m3x3ToCSR(DD_F3);
    free3x3(DD_F3);

    /* compute DDDD_F */
    temp51 = allocate5x5(S*S, S);
    temp52 = allocate5x5(S*S, S);
    temp53 = allocate5x5(S*S, S);
    temp54 = allocate5x5(S*S, S);
    DDDD_F5 = allocate5x5(S*S, S);

    temp2 = CSR_matrix_multiply(Dxy_1F, Dxy_2F);
    CSR_scalar_mult(temp2, 2.0);
    
    m3x3_matrix_multiply(Dxx_1F3, Dxx_2F3, temp51);
    csrTo5x5(temp2, temp52);
    m3x3_matrix_multiply(Dyy_1F3, Dyy_2F3, temp53);
    m5x5_matrix_add(temp51, temp52, temp54);
    m5x5_matrix_add(temp54, temp53, DDDD_F5);
    *DDDD_F = m5x5ToCSR(DDDD_F5);

    free5x5(DDDD_F5);

    CSR_free(temp2);

    /* free intermediate matrices */
    free3x3(Dxx_1F3);
    free3x3(Dxx_2F3);
    free3x3(Dyy_1F3);
    free3x3(Dyy_2F3);

    free5x5(temp51);
    free5x5(temp52);
    free5x5(temp53);
    free5x5(temp54);

    CSR_free(Dxx_1F);
    CSR_free(Dxx_2F);
    CSR_free(Dyy_1F);
    CSR_free(Dyy_2F);
    CSR_free(Dxy_1F);
    CSR_free(Dxy_2F);

    return 1;
}

/*
 * This is a C port of Alberto's Matlab function of the same name.
 * Currently it can only handle BC_flag 0 and BCVersion 1.
 * It also doesn't generate the DL_* matrices as the MP3D code doesn't use them, and
 * doesn't replace bits of DDDD_F on the edge of the plate with bits of identity matrix.
 */
int PlateEmbedded::myAiryRect(double nu, int Sx, int Sy, int BC_flag, int BCVersion,
			      CSRmatrix **DDDD_F, CSRmatrix **DD_F, CSRmatrix **Dxx_2F,
			      CSRmatrix **Dyy_2F)
{
    CSRmatrix *DxF, *Dxx_1F, *DyF, *Dyy_1F;
    double *diag1, *diag2;
    CSRmatrix *Dxy_1F, *Dxy_2F;
    int l, m;
    int I, J;
    CSRmatrix *CPcheck_F;
    CSRmatrix *temp1, *temp2, *temp3, *temp4, *temp5;

    /* generate Dxx and Dyy */
    diag1 = new double[Sx];
    memset(diag1, 0, Sx * sizeof(double));
    diag1[0] = -2.0;
    diag1[1] = 1.0;
    
    DxF = CSR_sym_toeplitz(diag1, Sx);
    Dxx_1F = CSR_kron_mat_eye(DxF, Sy);
    *Dxx_2F = CSR_duplicate(Dxx_1F);

    diag2 = new double[Sy];
    memset(diag2, 0, Sy * sizeof(double));
    diag2[0] = -2.0;
    diag2[1] = 1.0;
    
    DyF = CSR_sym_toeplitz(diag2, Sy);
    Dyy_1F = CSR_kron_eye_mat(DyF, Sx);
    *Dyy_2F = CSR_duplicate(Dyy_1F);

    delete[] diag1;
    delete[] diag2;
    CSR_free(DxF);
    CSR_free(DyF);

    /* generate Dxy */
    Dxy_2F = (CSRmatrix*)malloc(sizeof(CSRmatrix));
    CSR_setup(Dxy_2F, ((Sx-1)*(Sy-1)), Sx*Sy, ((Sx-1)*(Sy-1)*4));
    
    for (l = 0; l < (Sx-1); l++) {
	for (m = 0; m < (Sy-1); m++) {
	    I = m + (l * (Sy-1));
	    J = I + l;
	    CSRSetValue(Dxy_2F, I, J, 1.0);
	    CSRSetValue(Dxy_2F, I, J+1, -1.0);
	    CSRSetValue(Dxy_2F, I, J+Sy, -1.0);
	    CSRSetValue(Dxy_2F, I, J+Sy+1, 1.0);
	}
    }

    Dxy_1F = CSR_transpose(Dxy_2F);

    /* boundary conditions */
    for (m = 0; m < Sy; m++) {
	I = m;
	CSRSetValue(*Dxx_2F, I, I, -1.0);
	CSRSetValue(*Dxx_2F, I, I+Sy, 1.0);
	I = m + (Sx-1)*Sy;
	CSRSetValue(*Dxx_2F, I, I, -1.0);
	CSRSetValue(*Dxx_2F, I, I-Sy, 1.0);
    }

    for (l = 0; l < Sx; l++) {
	I = l * Sy;
	CSRSetValue(*Dyy_2F, I, I, -1.0);
	CSRSetValue(*Dyy_2F, I, I+1, 1.0);
	I = (l * Sy) + (Sy - 1);
	CSRSetValue(*Dyy_2F, I, I, -1.0);
	CSRSetValue(*Dyy_2F, I, I-1, 1.0);
    }

    for (l = 0; l < Sx; l++) {
	I = l * Sy;
	CSR_zero_row(Dxx_1F, I);
	CSR_zero_row(Dyy_1F, I);
	CSR_zero_row(Dxy_1F, I);
	CSR_zero_column(*Dxx_2F, I);
	CSR_zero_column(*Dyy_2F, I);
	CSR_zero_column(Dxy_2F, I);
	
	I = (l * Sy) + (Sy - 1);
	CSR_zero_row(Dxx_1F, I);
	CSR_zero_row(Dyy_1F, I);
	CSR_zero_row(Dxy_1F, I);
	CSR_zero_column(*Dxx_2F, I);
	CSR_zero_column(*Dyy_2F, I);
	CSR_zero_column(Dxy_2F, I);
    }

    for (m = 0; m < Sy; m++) {
	I = m;
	CSR_zero_row(Dxx_1F, I);
	CSR_zero_row(Dyy_1F, I);
	CSR_zero_row(Dxy_1F, I);
	CSR_zero_column(*Dxx_2F, I);
	CSR_zero_column(*Dyy_2F, I);
	CSR_zero_column(Dxy_2F, I);

	I = m + (Sx-1)*Sy;
	CSR_zero_row(Dxx_1F, I);
	CSR_zero_row(Dyy_1F, I);
	CSR_zero_row(Dxy_1F, I);
	CSR_zero_column(*Dxx_2F, I);
	CSR_zero_column(*Dyy_2F, I);
	CSR_zero_column(Dxy_2F, I);
    }

    /* scale matrices */
    CSR_scalar_mult(Dxx_1F, (1.0 / (h*h)));
    CSR_scalar_mult(Dyy_1F, (1.0 / (h*h)));
    CSR_scalar_mult(Dxy_1F, (1.0 / (h*h)));
    CSR_scalar_mult(*Dxx_2F, (1.0 / (h*h)));
    CSR_scalar_mult(*Dyy_2F, (1.0 / (h*h)));
    CSR_scalar_mult(Dxy_2F, (1.0 / (h*h)));    

    /* compute CPcheck_F */
    /* Dxx_1*Dyy_2 - 2*Dxy_1*Dxy_2 + Dyy1*Dxx3 */
    temp1 = CSR_matrix_multiply(Dxx_1F, *Dyy_2F);
    temp2 = CSR_matrix_multiply(Dxy_1F, Dxy_2F);
    CSR_scalar_mult(temp2, -2.0);
    temp3 = CSR_matrix_multiply(Dyy_1F, *Dxx_2F);
    temp4 = CSR_matrix_add(temp1, temp2);
    CPcheck_F = CSR_matrix_add(temp4, temp3);
    CSR_free(temp1);
    CSR_free(temp2);
    CSR_free(temp3);
    CSR_free(temp4);    

    /* compute DDDD_F */
    /* Dxx_1*Dxx_2 + 2*Dxy_1*Dxy_2 + Dyy_1*Dyy_3 + nu*CPcheck */
    temp1 = CSR_matrix_multiply(Dxx_1F, *Dxx_2F);
    temp2 = CSR_matrix_multiply(Dxy_1F, Dxy_2F);
    CSR_scalar_mult(temp2, 2.0);
    temp3 = CSR_matrix_multiply(Dyy_1F, *Dyy_2F);

    temp4 = CSR_matrix_add(temp1, temp2);
    if (CPcheck_F->rowStart[CPcheck_F->nrow] != 0) {
	/* csr library doesn't like matrices with no non-zeroes */
	CSR_scalar_mult(CPcheck_F, nu);
	temp5 = CSR_matrix_add(temp4, temp3);
	*DDDD_F = CSR_matrix_add(temp5, CPcheck_F);
	CSR_free(temp5);
    }
    else {
	*DDDD_F = CSR_matrix_add(temp4, temp3);
    }
    CSR_free(temp1);
    CSR_free(temp2);
    CSR_free(temp3);
    CSR_free(temp4);    

    /* apply bsxfun to DDDD_F */
    /*
     * The Matlab version uses bsxfun to replace parts of DDDD_F that correspond to the edges
     * of the plate with parts of the identity matrix instead. This is not needed for the
     * MP3D code as restricting the matrix rows/columns to TruePhi removes all the affected
     * bits, so it's not implemented yet.
     */

    /* compute DD_F */
    *DD_F = CSR_matrix_add(*Dxx_2F, *Dyy_2F);

    /* free intermediate matrices */
    CSR_free(Dxx_1F);
    CSR_free(Dxy_1F);
    CSR_free(Dyy_1F);
    CSR_free(Dxy_2F);

    return 1;
}

/*
 * This is a C port of Alberto's Matlab function of the same name.
 * Currently it can only handle BC_flag 0 and BCVersion 1.
 */
int PlateEmbedded::myBiharmRect(double nu, int Sx, int Sy, int BC_flag, int BCVersion,
				CSRmatrix **DDDD, CSRmatrix **DD, CSRmatrix **Dxx_2, CSRmatrix **Dyy_2,
				CSRmatrix **Dxy_2)
{
    CSRmatrix *Dx, *Dxx_1, *Dy, *Dyy_1;
    double *diag1, *diag2;
    CSRmatrix *Dxy_1;
    int l, m;
    int I, J;
    CSRmatrix *CPcheck;
    CSRmatrix *temp1, *temp2, *temp3, *temp4, *temp5;

    matrix_5x5_t *DDDD5, *CPcheck5;
    matrix_3x3_t *Dxx_13, *Dxx_23, *Dyy_13, *Dyy_23;
    matrix_3x3_t *DD3;
    matrix_5x5_t *temp51, *temp52, *temp53, *temp54, *temp55;

    /* generate Dxx and Dyy */
    diag1 = new double[Sx];
    memset(diag1, 0, Sx * sizeof(double));
    diag1[0] = -2.0;
    diag1[1] = 1.0;
    
    Dx = CSR_sym_toeplitz(diag1, Sx);
    Dxx_1 = CSR_kron_mat_eye(Dx, Sy);
    *Dxx_2 = CSR_duplicate(Dxx_1);

    diag2 = new double[Sy];
    memset(diag2, 0, Sy * sizeof(double));
    diag2[0] = -2.0;
    diag2[1] = 1.0;
    
    Dy = CSR_sym_toeplitz(diag2, Sy);
    Dyy_1 = CSR_kron_eye_mat(Dy, Sx);
    *Dyy_2 = CSR_duplicate(Dyy_1);

    delete[] diag1;
    delete[] diag2;
    CSR_free(Dx);
    CSR_free(Dy);

    /* generate Dxy */
    *Dxy_2 = (CSRmatrix*)malloc(sizeof(CSRmatrix));
    CSR_setup(*Dxy_2, Sx*Sy, Sx*Sy, ((Sx-1)*(Sy-1)*4));
    
    for (l = 0; l < (Sx-1); l++) {
	for (m = 0; m < (Sy-1); m++) {
	    I = m + (l * (Sy-1));
	    J = I + l;
	    CSRSetValue(*Dxy_2, I, J, 1.0);
	    CSRSetValue(*Dxy_2, I, J+1, -1.0);
	    CSRSetValue(*Dxy_2, I, J+Sy, -1.0);
	    CSRSetValue(*Dxy_2, I, J+Sy+1, 1.0);
	}
    }

    Dxy_1 = CSR_transpose(*Dxy_2);

    /* handle boundary conditions */
    if (BC_flag == 0) {
	for (m = 0; m < Sy; m++) {
	    I = m;
	    CSRSetValue(*Dxx_2, I, I, -1.0);
	    CSRSetValue(*Dxx_2, I, I+Sy, 1.0);
	    I = m + (Sx-1)*Sy;
	    CSRSetValue(*Dxx_2, I, I, -1.0);
	    CSRSetValue(*Dxx_2, I, I-Sy, 1.0);
	}

	for (l = 0; l < Sx; l++) {
	    I = l * Sy;
	    CSRSetValue(*Dyy_2, I, I, -1.0);
	    CSRSetValue(*Dyy_2, I, I+1, 1.0);
	    I = (l * Sy) + (Sy - 1);
	    CSRSetValue(*Dyy_2, I, I, -1.0);
	    CSRSetValue(*Dyy_2, I, I-1, 1.0);
	}

	for (l = 0; l < Sx; l++) {
	    I = l * Sy;
	    CSR_zero_row(Dxx_1, I);
	    CSR_zero_row(Dyy_1, I);
	    CSR_zero_row(Dxy_1, I);
	    CSR_zero_column(*Dxx_2, I);
	    CSR_zero_column(*Dyy_2, I);
	    CSR_zero_column(*Dxy_2, I);

	    I = (l * Sy) + (Sy - 1);
	    CSR_zero_row(Dxx_1, I);
	    CSR_zero_row(Dyy_1, I);
	    CSR_zero_row(Dxy_1, I);
	    CSR_zero_column(*Dxx_2, I);
	    CSR_zero_column(*Dyy_2, I);
	    CSR_zero_column(*Dxy_2, I);
	}

	for (m = 0; m < Sy; m++) {
	    I = m;
	    CSR_zero_row(Dxx_1, I);
	    CSR_zero_row(Dyy_1, I);
	    CSR_zero_row(Dxy_1, I);
	    CSR_zero_column(*Dxx_2, I);
	    CSR_zero_column(*Dyy_2, I);
	    CSR_zero_column(*Dxy_2, I);

	    I = m + (Sx-1)*Sy;
	    CSR_zero_row(Dxx_1, I);
	    CSR_zero_row(Dyy_1, I);
	    CSR_zero_row(Dxy_1, I);
	    CSR_zero_column(*Dxx_2, I);
	    CSR_zero_column(*Dyy_2, I);
	    CSR_zero_column(*Dxy_2, I);
	}
    }

    /* scale the matrices */
    CSR_scalar_mult(Dxx_1, (1.0 / (h*h)));
    CSR_scalar_mult(Dyy_1, (1.0 / (h*h)));
    CSR_scalar_mult(Dxy_1, (1.0 / (h*h)));
    CSR_scalar_mult(*Dxx_2, (1.0 / (h*h)));
    CSR_scalar_mult(*Dyy_2, (1.0 / (h*h)));
    CSR_scalar_mult(*Dxy_2, (1.0 / (h*h)));

    /* convert matrices to 3x3 form */
    Dxx_13 = allocate3x3(Sx*Sy, Sy);
    Dyy_13 = allocate3x3(Sx*Sy, Sy);
    Dxx_23 = allocate3x3(Sx*Sy, Sy);
    Dyy_23 = allocate3x3(Sx*Sy, Sy);
    csrTo3x3(Dxx_1, Dxx_13);
    csrTo3x3(Dyy_1, Dyy_13);
    csrTo3x3(*Dxx_2, Dxx_23);
    csrTo3x3(*Dyy_2, Dyy_23);

    /* banded version of DD computation */
    DD3 = allocate3x3(Sx*Sy, Sy);
    m3x3_matrix_add(Dxx_23, Dyy_23, DD3);
    *DD = m3x3ToCSR(DD3);
    free3x3(DD3);

    /* compute CPcheck */
    /* Dxx_1*Dyy_2 - 2*Dxy_1*Dxy_2 + Dyy1*Dxx3 */
    temp2 = CSR_matrix_multiply(Dxy_1, *Dxy_2);
    CSR_scalar_mult(temp2, -2.0);

    /* banded version of CPcheck computation */
    /* allocate matrices first */
    temp51 = allocate5x5(Sx*Sy, Sy);
    temp52 = allocate5x5(Sx*Sy, Sy);
    temp53 = allocate5x5(Sx*Sy, Sy);
    temp54 = allocate5x5(Sx*Sy, Sy);
    temp55 = allocate5x5(Sx*Sy, Sy);
    CPcheck5 = allocate5x5(Sx*Sy, Sy);

    m3x3_matrix_multiply(Dxx_13, Dyy_23, temp51);
    /* Dxy_1 and Dxy_2 aren't banded right here so use CSR version */
    csrTo5x5(temp2, temp52);
    m3x3_matrix_multiply(Dyy_13, Dxx_23, temp53);
    m5x5_matrix_add(temp51, temp52, temp54);
    m5x5_matrix_add(temp53, temp54, CPcheck5);

    CSR_free(temp2);

    /* banded version of DDDD computation */
    DDDD5 = allocate5x5(Sx*Sy, Sy);
    m3x3_matrix_multiply(Dxx_13, Dxx_23, temp51);
    /* temp52 just needs negated */
    m5x5_scalar_mult(temp52, -1.0);
    m3x3_matrix_multiply(Dyy_13, Dyy_23, temp53);
    m5x5_matrix_add(temp51, temp52, temp54);
    m5x5_scalar_mult(CPcheck5, nu);
    m5x5_matrix_add(temp54, temp53, temp55);
    m5x5_matrix_add(temp55, CPcheck5, DDDD5);
    *DDDD = m5x5ToCSR(DDDD5);
    free5x5(DDDD5);
    free5x5(CPcheck5);

    free3x3(Dxx_13);
    free3x3(Dxx_23);
    free3x3(Dyy_13);
    free3x3(Dyy_23);

    free5x5(temp51);
    free5x5(temp52);
    free5x5(temp53);
    free5x5(temp54);
    free5x5(temp55);

    /* free intermediate matrices */
    CSR_free(Dxx_1);
    CSR_free(Dyy_1);
    CSR_free(Dxy_1);

    return 1;
}
 
double *PlateEmbedded::getEnergy()
{
    return energy;
}
