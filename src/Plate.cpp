/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

extern "C" {
#include "matgen.h"
};

#include "Plate.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "Input.h"
#include "SettingsManager.h"
#include "MathUtil.h"

#include <cmath>
#include <cstring>
using namespace std;

Plate::Plate(string name, Material *material, double thickness, double tension,
	     double lx, double ly, double t60_0, double t60_1000, int bc) : Component2D(name)
{
    double rho = material->getDensity();
    double E = material->getYoungsModulus();
    double nu = material->getPoissonsRatio();
    init(nu, rho, E, thickness, tension, lx, ly, t60_0, t60_1000, bc);
}

Plate::Plate(string name, double nu, double rho, double E, double thickness, double tension,
	     double lx, double ly, double t60_0, double t60_1000, int bc) : Component2D(name)
{
    init(nu, rho, E, thickness, tension, lx, ly, t60_0, t60_1000, bc);    
}

void Plate::init(double nu, double rho, double E, double thickness, double tension,
		 double lx, double ly, double t60_0, double t60_1000, int bc)
{
    int i;
    // compute flexural rigidity
    d0 = E * (thickness*thickness*thickness) /
	(12.0 * (1.0 - (nu*nu)));
    logMessage(1, "Flexural rigidity: %f", d0);
    h0 = thickness;
    t0 = tension;

    // stiffness parameter
    kappa = sqrt(d0 / (rho * thickness));

    // wave speed
    c = sqrt(tension / (rho * thickness));

    logMessage(1, "Kappa: %f, c: %f", kappa, c);

    GlobalSettings *gs = GlobalSettings::getInstance();
    SettingsManager *sm = SettingsManager::getInstance();

    sig_0 = (6.0 * log(10.0)) / t60_0;
    double z1 = 0.0;
    if (c == 0.0) {
	z1 = (2.0 * M_PI * 1000.0) / kappa;
    }
    else if (kappa == 0.0) {
	z1 = ((2.0 * M_PI * 1000.0) * (2.0 * M_PI * 1000.0)) / (c*c);
    }
    else {
	z1 = (-(c*c) + sqrt((c*c*c*c) +
			    (4.0*kappa*kappa*(2.0*M_PI*1000.0)*(2.0*M_PI*1000.0)))) /
	    (2*kappa*kappa);
    }
    sig_1 = ((6.0 * log(10.0)) / z1) * ((1.0 / t60_1000) - (1.0 / t60_0));


    // sig_0 and sig_1 need to be zeroed if loss is off
    switch (sm->getIntSetting(name, "loss_mode")) {
    case 0:
	sig_0 = 0.0;
	sig_1 = 0.0;
	break;
    case -1:
	sig_0 = 0.0;
	break;
    }

    logMessage(1, "sig_0: %f, sig_1: %f", sig_0, sig_1);

    double k = gs->getK();

    // grid spacing
    h = sqrt((c*c*k*k) + (4.0*sig_1*k) +
	     sqrt((c*c*k*k + 4.0*sig_1*k)*(c*c*k*k + 4.0*sig_1*k) +
		  16.0*kappa*kappa*k*k));
    h = h * sm->getDoubleSetting(name, "fixpar");

    int Nx = (int)floor(lx / h);
    h = lx / (double)Nx;
    int Ny = (int)floor(ly / h);
    mu = (k * kappa) / (h*h);
    lambda = (c*k) / h;

    logMessage(1, "h: %f, Nx: %d, Ny: %d, mu: %f, lambda: %f", h, Nx, Ny, mu, lambda);

    allocateState(Nx+1, Ny+1);

    alpha = (k*k) / (rho * h0 * (1.0 + k*sig_0) * h * h);
    bowFactor = k / (2.0 * (1.0 + k*sig_0) * h * h);

    CSRmatrix *DD, *D, *D2, *D3, *Dxy2, *Dx, *Dy;
    double *scalevec, *scalevecx, *scalevecy;
    CSRmatrix *tm, *tm2, *tm3;

    plateMatGen(Nx, Ny, bc, nu, &DD, &D, &D2, &D3, &Dxy2, &scalevec, &scalevecx, &scalevecy, &Dx, &Dy);

    /* compute update matrix B */
    tm = CSR_create_eye(ss);
    CSR_scalar_mult(tm, 2.0);
    CSR_scalar_mult(DD, (mu*mu));
    tm2 = CSR_matrix_add_sub(tm, DD, 1);
    CSR_free(tm);

    tm = CSR_duplicate(D);
    CSR_scalar_mult(tm, (lambda*lambda));
    tm3 = CSR_matrix_add_sub(tm2, tm, 0);
    CSR_free(tm2);
    CSR_free(tm);
    
    tm = CSR_duplicate(D);
    CSR_scalar_mult(tm, ((2.0*sig_1*k) / (h*h)));
    B = CSR_matrix_add_sub(tm3, tm, 0);
    CSR_free(tm3);
    CSR_free(tm);

    CSR_scalar_computation(B, (1.0 + (sig_0*k)), 4);

    /* compute update matrix C */
    tm = CSR_create_eye(ss);
    CSR_scalar_mult(tm, ((sig_0*k) - 1.0));
    CSR_scalar_mult(D, ((2.0*sig_1*k)/(h*h)));
    C = CSR_matrix_add_sub(tm, D, 1);
    CSR_free(tm);

    CSR_scalar_computation(C, (1.0 + (sig_0*k)), 4);

    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps()];

	D2_mat = D2;
	CSR_scalar_mult(D2_mat, -1.0);
	D3_mat = D3;
	CSR_scalar_mult(D3_mat, -1.0);
	Dxy2_mat = Dxy2;
	CSR_scalar_mult(Dxy2_mat, -(1.0 - nu));
	Dx_mat = Dx;
	Dy_mat = Dy;

	scalevec_t = new double[(Nx+1)*(Ny+1)];
	scalevec_mxmy = new double[(Nx+1)*(Ny+1)];
	scalevec_mxmy2 = new double[(Nx+1)*(Ny+1)];
	scalevec_mxy = new double[Nx*Ny];
	scalevec_x_total = new double[Nx*(Ny+1)];
	scalevec_y_total = new double[(Nx+1)*Ny];

	for (i = 0; i < ((Nx+1)*(Ny+1)); i++) {
	    scalevec_t[i] = 0.5 * rho * thickness * (h*h) * scalevec[i] / (k*k);
	    scalevec_mxmy[i] = 0.5 * d0 * scalevec[i] / ((1.0 - (nu*nu))*(h*h));
	    scalevec_mxmy2[i] = 0.5 * d0 * scalevec[i] * nu / ((1.0 - (nu*nu))*(h*h));
	}
	for (i = 0; i < (Nx*Ny); i++) {
	    scalevec_mxy[i] = d0 / ((1.0 - nu) * (h*h));
	}
	for (i = 0; i < (Nx*(Ny+1)); i++) {
	    scalevec_x_total[i] = 0.5 * tension * scalevecx[i];
	}
	for (i = 0; i < ((Nx+1)*Ny); i++) {
	    scalevec_y_total[i] = 0.5 * tension * scalevecy[i];
	}

	mx = new double[(Nx+1)*(Ny+1)];
	mx1 = new double[(Nx+1)*(Ny+1)];
	my = new double[(Nx+1)*(Ny+1)];
	my1 = new double[(Nx+1)*(Ny+1)];
	mxy = new double[Nx*Ny];
	mxy1 = new double[Nx*Ny];

	Dxu = new double[Nx*(Ny+1)];
	Dxu1 = new double[Nx*(Ny+1)];
	Dyu = new double[(Nx+1)*Ny];
	Dyu1 = new double[(Nx+1)*Ny];
    }
    else {
	energy = NULL;
	CSR_free(D2);
	CSR_free(D3);
	CSR_free(Dxy2);  
	CSR_free(Dx);  
	CSR_free(Dy);  
    }

    CSR_free(DD);
    CSR_free(D);

    free(scalevec);
    free(scalevecx);
    free(scalevecy);

#ifdef USE_GPU
    gpuPlate = NULL;
#endif
}

void Plate::logMatrices()
{
    saveMatrix(B, "B");
    saveMatrix(C, "C");
}

Plate::~Plate()
{
    CSR_free(B);
    CSR_free(C);

    if (energy) {
	delete[] energy;

	CSR_free(D2_mat);
	CSR_free(D3_mat);
	CSR_free(Dxy2_mat);
	CSR_free(Dx_mat);
	CSR_free(Dy_mat);

	delete[] scalevec_t;
	delete[] scalevec_mxmy;
	delete[] scalevec_mxmy2;
	delete[] scalevec_mxy;
	delete[] scalevec_x_total;
	delete[] scalevec_y_total;

	delete[] mx;
	delete[] mx1;
	delete[] my;
	delete[] my1;
	delete[] mxy;
	delete[] mxy1;

	delete[] Dxu;
	delete[] Dxu1;
	delete[] Dyu;
	delete[] Dyu1;
    }

#ifdef USE_GPU
    if (gpuPlate) {
	delete gpuPlate;
	// state arrays will be freed by gpuPlate, don't let Component destructor free them
	u = NULL;
	u1 = NULL;
	u2 = NULL;
    }
#endif
}


void Plate::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuPlate != NULL) {
	gpuPlate->runTimestep(n, u, u1, u2);

	// remember halo
	runInputs(n, &u[ny+ny], &u1[ny+ny], &u2[ny+ny]);
	return;
    }
#endif
    CSR_matrix_vector(B, u1, u, 1, 1); // u = B*u1
    CSR_matrix_vector(C, u2, u, 0, 1); // u = C*u2 + u

    runInputs(n, u, u1, u2);
}

void Plate::swapBuffers(int n)
{
    if (energy) {
	int i;
	int Nx = nx-1; // match the Nx and Ny in Matlab to avoid confusion when porting
	int Ny = ny-1;

	// already negated D2_mat, D3_mat and Dxy2_mat so no need to negate results here
	CSR_matrix_vector_mult(D2_mat, u, mx);
	CSR_matrix_vector_mult(D2_mat, u1, mx1);
	CSR_matrix_vector_mult(D3_mat, u, my);
	CSR_matrix_vector_mult(D3_mat, u1, my1);
	CSR_matrix_vector_mult(Dxy2_mat, u, mxy);
	CSR_matrix_vector_mult(Dxy2_mat, u1, mxy1);

	CSR_matrix_vector_mult(Dx_mat, u, Dxu);
	CSR_matrix_vector_mult(Dx_mat, u1, Dxu1);
	CSR_matrix_vector_mult(Dy_mat, u, Dyu);
	CSR_matrix_vector_mult(Dy_mat, u1, Dyu1);

	double etot = 0.0;
	for (i = 0; i < ((Nx+1)*(Ny+1)); i++) {
	    // T(n) term
	    etot += scalevec_t[i] * ((u[i]-u1[i])*(u[i]-u1[i]));

	    // Vp(n) term (except for scalevec_mxy part)
	    etot += scalevec_mxmy[i] * ((mx[i]*mx1[i])+(my[i]*my1[i]));
	    etot -= scalevec_mxmy2[i] * ((mx[i]*my1[i])+(my[i]*mx1[i]));
	}
	for (i = 0; i < (Nx*Ny); i++) {
	    // rest of Vp(n)
	    etot += scalevec_mxy[i] * mxy[i] * mxy1[i];
	}
	// Vm(n) term
	for (i = 0; i < (Nx*(Ny+1)); i++) {
	    etot += scalevec_x_total[i] * Dxu[i] * Dxu1[i];
	}
	for (i = 0; i < ((Nx+1)*Ny); i++) {
	    etot += scalevec_y_total[i] * Dyu[i] * Dyu1[i];
	}
	energy[n] = etot;
    }

    Component::swapBuffers(n);
}

int Plate::getGPUScore()
{
    if (SettingsManager::getInstance()->getBoolSetting(name, "disable_gpu")) return GPU_SCORE_NO;
    return GPU_SCORE_GOOD;
}

int Plate::getGPUMemRequired()
{
    return ((ss + 4) * 3 * sizeof(double)) + // state arrays
	(ss * 2) + // index arrays
	(13 * 256 * 2 * sizeof(double)); // max coefficients
}

bool Plate::moveToGPU()
{
#ifdef USE_GPU
    // try to move to GPU
    double *d_u, *d_u1, *d_u2;
    int i;
    gpuPlate = new GPUPlate(nx, ny, B, C, &d_u, &d_u1, &d_u2);
    if (!gpuPlate->isOK()) {
	// didn't work, keep on host
	logMessage(1, "Moving plate %s to GPU failed", name.c_str());
	delete gpuPlate;
	gpuPlate = NULL;
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
    logMessage(1, "Plate %s moved to GPU", name.c_str());
    return true;
#else
    return false;
#endif
}

double *Plate::getEnergy()
{
    return energy;
}
