/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "Embedding.h"
#include "GlobalSettings.h"
#include "Logger.h"

#include <cstring>
#include <cmath>
using namespace std;

Embedding::Embedding(Airbox *airbox, PlateEmbedded *plate)
    : Connection(airbox, plate)
{
    logMessage(1, "Creating Embedding of %s within %s", plate->getName().c_str(),
	       airbox->getName().c_str());

    // get required values from plate
    int plateSs = plate->getStateSize();
    double cx = plate->getCx();
    double cy = plate->getCy();
    double cz = plate->getCz();
    int Nx, Ny;
    bool isCircular = plate->getCircular();
    double h = plate->getH();
    double H = plate->getThickness();
    double k = plate->getK();
    double rho = plate->getRho();
    double a0 = plate->getA0();
    int TruePhiSize = plate->getTruePhiSize();
    int *TruePhi = plate->getTruePhi();
    double lx = plate->getLx();

    // get back the original Nx and Ny values
    if (isCircular) {
	Nx = plate->getNx() - 1;
	Ny = plate->getNy() - 1;
    }
    else {
	Nx = plate->getNx() + 1;
	Ny = plate->getNy() + 1;
    }
    int SS = (Nx+1) * (Ny+1);

    // get required values from airbox
    double LZ = airbox->getLZ();
    double Q = airbox->getQ();
    double rhoA = airbox->getRhoA();
    double Gamma = airbox->getGamma();
    int anx = airbox->getNx();
    int any = airbox->getNy();

    int true_Psi_count;
    int i, j, jdx;
    double gridx, gridy, xstart, ystart, xend, yend;

    double SR = GlobalSettings::getInstance()->getSampleRate();

    // allocate buffers
    Diff_size = airbox->getNx() * airbox->getNy();
    Diff_np = new double[Diff_size];
    Diff_n = new double[Diff_size];
    Diff_nm = new double[Diff_size];
    Sum_np = new double[Diff_size];
    Diff_tmp = new double[Diff_size];
    true_Psi = new double[Diff_size];
    Diffstar_n = new double[Diff_size];
    Diffstar_nm = new double[Diff_size];

    transferBuffer = new double[plateSs];

    // clear buffers
    memset(Diff_np, 0, Diff_size * sizeof(double));
    memset(Diff_n, 0, Diff_size * sizeof(double));
    memset(Diff_nm, 0, Diff_size * sizeof(double));
    memset(Sum_np, 0, Diff_size * sizeof(double));
    memset(Diff_tmp, 0, Diff_size * sizeof(double));
    memset(true_Psi, 0, Diff_size * sizeof(double));
    memset(Diffstar_n, 0, Diff_size * sizeof(double));
    memset(Diffstar_nm, 0, Diff_size * sizeof(double));

    memset(transferBuffer, 0, plateSs * sizeof(double));

    // compute pd and pu
    pd = ((int)floor((cz + LZ / 2.0) / Q)) - 1;
    pu = pd + 1;
    logMessage(1, "pd: %d, pu: %d", pd, pu);

    // compute Bf
    Bf = (k * rhoA) / (2.0 * (1.0 + a0) * rho * H);

    // compute true_Psi
    // this is a mask the size of a layer of airbox that contains 1 for points
    // within the plate and 0 for points outside it
    xstart = cx - ((double)Nx / 2.0) * h;
    xend = cx + ((double)Nx / 2.0) * h;
    ystart = cy - ((double)Ny / 2.0) * h;
    yend = cy + ((double)Ny / 2.0) * h;

    if (!isCircular) {
	jdx = 0;
	true_Psi_count = 0;
	gridx = (((double)-(anx-1)) / 2.0) * Q;
	for (i = 0; i < anx; i++) {
	    gridy = (((double)-(any-1)) / 2.0) * Q;
	    for (j = 0; j < any; j++) {
		true_Psi[jdx] = 0.0;
		if ((gridx >= xstart) && (gridx <= xend) &&
		    (gridy >= ystart) && (gridy <= yend)) {
		    true_Psi[jdx] = 1.0;
		    true_Psi_count++;
		}
		jdx++;
		gridy += Q;
	    }
	    gridx += Q;
	}
    }
    else {
	double radsq = ((lx / 2.0) * (lx / 2.0));
	jdx = 0;
	true_Psi_count = 0;
	gridx = (((double)-(anx-1)) / 2.0) * Q;
	for (i = 0; i < anx; i++) {
	    gridy = (((double)-(any-1)) / 2.0) * Q;
	    for (j = 0; j < any; j++) {
		true_Psi[jdx] = 0.0;

		if ((((gridx-cx)*(gridx-cx))+((gridy-cy)*(gridy-cy))) < radsq) {
		    true_Psi[jdx] = 1.0;
		    true_Psi_count++;
		}
		
		jdx++;
		gridy += Q;
	    }
	    gridx += Q;
	}
	logMessage(1, "lx = %f, radsq = %f, true_Psi_count = %d", lx, radsq, true_Psi_count);
    }

    // compute IMat and JMat
    // need plate h, cx, cy, Nx, Ny, h (for xstart and ystart as well)
    // need airbox nx, ny, Q
    if (h > Q) {
	// JMat first 
	JMat = (CSRmatrix *)malloc(sizeof(CSRmatrix));
	CSR_setup(JMat, Diff_size, SS, true_Psi_count*4);
	jdx = 0;
	gridx = (((double)-(anx-1)) / 2.0) * Q;
	for (i = 0; i < anx; i++) {
	    gridy = (((double)-(any-1)) / 2.0) * Q;
	    for (j = 0; j < any; j++) {
		if (true_Psi[jdx]) {
		    int l0, m0;
		    double alpha, beta;
		    int J;
		    
		    l0 = (int)floor((gridx - xstart) / h);
		    m0 = (int)floor((gridy - ystart) / h);
		    alpha = ((gridx - xstart) / h) - (double)l0;
		    beta = ((gridy - ystart) / h) - (double)m0;
		    J = m0 + (l0 * (Ny+1));
		    
		    CSRSetValue(JMat, jdx, J, (1.0-alpha)*(1.0-beta));
		    CSRSetValue(JMat, jdx, J+1, (1.0-alpha)*(beta));
		    CSRSetValue(JMat, jdx, J+Ny+1, (alpha)*(1.0-beta));
		    CSRSetValue(JMat, jdx, J+Ny+2, (alpha)*(beta));
		}
		
		jdx++;
		gridy += Q;
	    }
	    gridx += Q;
	}
	
	/* IMat is based on transpose of JMat */
	IMat = CSR_transpose(JMat);
	CSR_scalar_mult(IMat, (Q*Q)/(h*h));
    }
    else {
	// IMat first
	logMessage(1, "Generating IMat first");
	IMat = (CSRmatrix *)malloc(sizeof(CSRmatrix));
	CSR_setup(IMat, SS, Diff_size, true_Psi_count*4);
	jdx = 0;
	xstart = (((double)-(anx-1)) / 2.0) * Q;
	ystart = (((double)-(any-1)) / 2.0) * Q;
	gridx = (((double)-Nx) / 2.0) * h + cx;
	for (i = 0; i < (Nx+1); i++) {
	    gridy = (((double)-Ny) / 2.0) * h + cy;
	    for (j = 0; j < (Ny+1); j++) {
		int l0, m0;
		double alpha, beta;
		int J;

		l0 = (int)floor((gridx - xstart) / Q);
		m0 = (int)floor((gridy - ystart) / Q);
		alpha = ((gridx - xstart) / Q) - (double)l0;
		beta = ((gridy - ystart) / Q) - (double)m0;
		J = m0 + (l0 * any);

		if (true_Psi[J]) CSRSetValue(IMat, jdx, J, (1.0-alpha)*(1.0-beta));
		if (true_Psi[J+1]) CSRSetValue(IMat, jdx, J+1, (1.0-alpha)*(beta));
		if (true_Psi[J+any]) CSRSetValue(IMat, jdx, J+any, (alpha)*(1.0-beta));
		if (true_Psi[J+any+1]) CSRSetValue(IMat, jdx, J+any+1, (alpha)*(beta));

		jdx++;
		gridy += h;
	    }
	    gridx += h;
	}
	/* the NewMask multiply in the Matlab is factored into the loop above */

	/* JMat is based on transpose of IMat */
	JMat = CSR_transpose(IMat);
	CSR_scalar_mult(JMat, (h*h)/(Q*Q));	
    }

    // crop/zero IMat and JMat down
    if (!isCircular) {
	CSR_cut_cols(JMat, TruePhi, TruePhiSize);
	CSR_cut_rows(IMat, TruePhi, TruePhiSize);
    }
    else {
	CSR_zero_cols(JMat, TruePhi, TruePhiSize);
	CSR_zero_rows(IMat, TruePhi, TruePhiSize);
    }

    // pre-scale IMat
    BfIMat = CSR_duplicate(IMat);
    CSR_scalar_mult(BfIMat, Bf);

    // pass IMat, JMat and transferBuffer to the plate
    plate->setInterpolation(IMat, JMat, transferBuffer, Q * SR * Gamma * Gamma * Bf);

    // inform the airbox that there's a plate here
    airbox->addPlate(pd, true_Psi);

#ifdef USE_GPU
    gpuEmbedding = NULL;
#endif
}

Embedding::~Embedding()
{
    CSR_free(IMat);
    CSR_free(JMat);
    CSR_free(BfIMat);
    
    delete[] Diff_np;
    delete[] Diff_n;
    delete[] Diff_nm;
    delete[] Sum_np;
    delete[] Diff_tmp;
    delete[] true_Psi;
    delete[] transferBuffer;
    delete[] Diffstar_n;
    delete[] Diffstar_nm;

#ifdef USE_GPU
    if (gpuEmbedding) delete gpuEmbedding;
#endif
}

void Embedding::runTimestep(int n)
{
    // casts are safe, we know the types since we passed them in
    // to the constructor
    PlateEmbedded *plate = (PlateEmbedded *)c2;
    Airbox *airbox = (Airbox *)c1;

#ifdef USE_GPU
    if (gpuEmbedding) {
	gpuEmbedding->runTimestep(n, airbox, transferBuffer);
	return;
    }
#endif

    double *tmp;
    double *psiup, *psidown;
    int i;

    double Gamma = airbox->getGamma();
    double lambda = airbox->getLambda();
    double Q = airbox->getQ();
    double k = plate->getK();

    // reverse interpolation from plate back to airbox
    // remember we have to use u1 here because it needs to get included
    // in the partial update
    psidown = airbox->getU1() + pd * Diff_size;
    psiup = airbox->getU1() + pu * Diff_size;
    // plate should have placed w - w0 (u - u2) in transferBuffer for us
    CSR_matrix_vector_mult(JMat, transferBuffer, Diff_tmp);
    for (i = 0; i < Diff_size; i++) {
	Diffstar_n[i] = -(true_Psi[i]*Diff_n[i]) + ((Q/k)*Diff_tmp[i]);
	Diff_np[i] = Diff_np[i] - (Gamma*Gamma*Diffstar_n[i]) +
	    (Gamma*Gamma*true_Psi[i]*Diff_n[i]);
	/*Diff_np[i] = Diff_np[i] - ((Q/k)*Gamma*Gamma*Diff_tmp[i]) +
	  (2.0*Gamma*Gamma*true_Psi[i]*Diff_n[i]);*/

	if (true_Psi[i] > 0.5) {
	    psidown[i] = 0.5 * (Sum_np[i] + Diff_np[i]);
	    psiup[i] = 0.5 * (Sum_np[i] - Diff_np[i]);
	}
    }

    // re-run slice of airbox
    airbox->runPartialUpdate(pd - 1, 4);

    // swap buffers
    tmp = Diff_nm;
    Diff_nm = Diff_n;
    Diff_n = Diff_np;
    Diff_np = tmp;

    tmp = Diffstar_nm;
    Diffstar_nm = Diffstar_n;
    Diffstar_n = tmp;

    // forward interpolation from airbox to plate
    psidown = airbox->getU() + pd * Diff_size;
    psiup = airbox->getU() + pu * Diff_size;
    double diffnfac = 2.0 * Gamma * Gamma;
    for (i = 0; i < Diff_size; i++) {
	Sum_np[i] = psidown[i] + psiup[i];
	Diff_np[i] = psidown[i] - psiup[i];

	Diff_np[i] += lambda*lambda*(Diffstar_nm[i] - Diff_nm[i]);

	Diff_tmp[i] = Diff_np[i] + (diffnfac * Diff_n[i]) - Diff_nm[i];
    }
    CSR_matrix_vector_mult(BfIMat, Diff_tmp, transferBuffer);
}

void Embedding::maybeMoveToGPU()
{
#ifdef USE_GPU
    if (c1->isOnGPU()) {
	PlateEmbedded *plate = (PlateEmbedded *)c2;
	Airbox *airbox = (Airbox *)c1;

	// airbox is on GPU - move embedding to GPU as well
	// embedded plates can't currently be on GPU so will be on host
	gpuEmbedding = new GPUEmbedding(BfIMat, JMat, Diff_size, pd, pu, airbox,
					plate->getK(), plate->getStateSize(),
					true_Psi);
	logMessage(1, "Moved embedding to GPU");
    }
#endif
}
