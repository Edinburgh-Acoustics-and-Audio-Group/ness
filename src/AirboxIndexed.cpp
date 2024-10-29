/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

/*
 * This class models a 3D box of air, optionally with things (plates, membranes,
 * drum shells) embedded within it.
 *
 * To save memory bandwidth, it avoids using a full update matrix, which would be
 * inefficient as the co-efficients are the same across most of the box. Instead,
 * it stores a number of co-efficient sets (in the CUDA version these are kept in
 * fast constant memory on the GPU) and uses an index array containing a single
 * byte per grid point to indicate which set should be used.
 *
 * Normally, the same co-efficients are used across the entire interior of the box
 * and only the boundaries are different. However, a vertical cylindrical drum
 * shell may be added to the box (see addDrumShell below), and this will cause
 * different co-efficients to be used for the points adjacent to the shell.
 *
 * The box doesn't need to know if there are plates and membranes embedded in it,
 * as this is all handled by the Embedding object. (Except when the energy
 * conservation check is enabled).
 */

#include "AirboxIndexed.h"
#include "Logger.h"
#include "Input.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"

#include <cmath>
#include <cstring>
using namespace std;

AirboxIndexed::AirboxIndexed(string name, double lx, double ly, double lz, double c_a, double rho_a)
    : Airbox(name)
{
    setup(lx, ly, lz, c_a, rho_a);

    int i;

    /* calculate update co-efficients */
    APSI_CORNER = 1.0 / (1.0 + 3.0*Gamma);
    APSI_EDGE = 1.0 / (1.0 + 2.0*Gamma);
    APSI_FACE = 1.0 / (1.0 + Gamma);
    
    CPSI_CORNER = 1.0 - 3.0*Gamma;
    CPSI_EDGE = 1.0 - 2.0*Gamma;
    CPSI_FACE = 1.0 - Gamma;

    coeff0 = 2.0 - (6.0*Gamma*Gamma);
    coeff1 = Gamma*Gamma;
    coeff2 = 2.0 * coeff1;

    th = 6e-6;

    if (SettingsManager::getInstance()->getIntSetting(name, "loss_mode") == 0) {
	th = 0.0;
    }

    GlobalSettings *gs = GlobalSettings::getInstance();
    energy = NULL;
    reflection = false;
    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps() + 1];
	energyMask = new unsigned char[ss];
	reflection = true; // need this to get energy conservation
    }

    double lossfac = th * c_a * k / (Q*Q);

    // generate index array and coefficient sets
    coeffs = new double[16 * 256]; // only 9 needed per set but this is better for alignment
    index = new unsigned char[ss];

    memset(coeffs, 0, 16*256*sizeof(double));

    int x, y, z;
    int idx = 0;
    for (z = 0; z < nz; z++) {
	for (x = 0; x < nx; x++) {
	    for (y = 0; y < ny; y++) {
		if (energy) {
		    // bits 0, 1 and 2 of energyMask indicate whether to include this point in the
		    // X, Y and Z energy tallies respectively. 1 means yes (true for most points),
		    // 0 means no
		    energyMask[idx] = 7;
		    if (x == (nx-1)) energyMask[idx] &= 0xfe;
		    if (y == (ny-1)) energyMask[idx] &= 0xfd;
		    if (z == (nz-1)) energyMask[idx] &= 0xfb;
		}

		if (z == 0) {
		    if (y == 0) {
			if (x == 0) {
			    index[idx] = 0;
			}
			else if (x < (nx-1)) {
			    index[idx] = 1;
			}
			else {
			    index[idx] = 2;
			}
		    }
		    else if (y < (ny-1)) {
			if (x == 0) {
			    index[idx] = 3;
			}
			else if (x < (nx-1)) {
			    index[idx] = 4;
			}
			else {
			    index[idx] = 5;
			}
		    }
		    else {
			if (x == 0) {
			    index[idx] = 6;
			}
			else if (x < (nx-1)) {
			    index[idx] = 7;
			}
			else {
			    index[idx] = 8;
			}
		    }
		}
		else if (z < (nz-1)) {
		    if (y == 0) {
			if (x == 0) {
			    index[idx] = 9;
			}
			else if (x < (nx-1)) {
			    index[idx] = 10;
			}
			else {
			    index[idx] = 11;
			}
		    }
		    else if (y < (ny-1)) {
			if (x == 0) {
			    index[idx] = 12;
			}
			else if (x < (nx-1)) {
			    index[idx] = 13;
			}
			else {
			    index[idx] = 14;
			}
		    }
		    else {
			if (x == 0) {
			    index[idx] = 15;
			}
			else if (x < (nx-1)) {
			    index[idx] = 16;
			}
			else {
			    index[idx] = 17;
			}
		    }
		}
		else {
		    if (y == 0) {
			if (x == 0) {
			    index[idx] = 18;
			}
			else if (x < (nx-1)) {
			    index[idx] = 19;
			}
			else {
			    index[idx] = 20;
			}
		    }
		    else if (y < (ny-1)) {
			if (x == 0) {
			    index[idx] = 21;
			}
			else if (x < (nx-1)) {
			    index[idx] = 22;
			}
			else {
			    index[idx] = 23;
			}
		    }
		    else {
			if (x == 0) {
			    index[idx] = 24;
			}
			else if (x < (nx-1)) {
			    index[idx] = 25;
			}
			else {
			    index[idx] = 26;
			}
		    }
		}
		idx++;
	    }
	}
    }

    // generate co-efficient sets
    idx = 0;
    for (z = 0; z < 3; z++) {
	for (y = 0; y < 3; y++) {
	    for (x = 0; x < 3; x++) {
		int xtrem = 0;
		coeffs[(idx << 4) + 0] = coeff0 - 6.0*lossfac; /* centre point */
		if (x == 0) {
		    xtrem++;
		    coeffs[(idx << 4) + 1] = coeff2 + 2.0*lossfac;
		    coeffs[(idx << 4) + 9] = 2.0*lossfac;
		}
		else if (x == 1) {
		    coeffs[(idx << 4) + 1] = coeff1 + lossfac;
		    coeffs[(idx << 4) + 9] = lossfac;
		    coeffs[(idx << 4) + 2] = coeff1 + lossfac;
		    coeffs[(idx << 4) + 10] = lossfac;
		}
		else {
		    xtrem++;
		    coeffs[(idx << 4) + 2] = coeff2 + 2.0*lossfac;
		    coeffs[(idx << 4) + 10] = 2.0*lossfac;
		}

		if (y == 0) {
		    xtrem++;
		    coeffs[(idx << 4) + 3] = coeff2 + 2.0*lossfac;
		    coeffs[(idx << 4) + 11] = 2.0*lossfac;
		}
		else if (y == 1) {
		    coeffs[(idx << 4) + 3] = coeff1 + lossfac;
		    coeffs[(idx << 4) + 11] = lossfac;
		    coeffs[(idx << 4) + 4] = coeff1 + lossfac;
		    coeffs[(idx << 4) + 12] = lossfac;
		}
		else {
		    xtrem++;
		    coeffs[(idx << 4) + 4] = coeff2 + 2.0*lossfac;
		    coeffs[(idx << 4) + 12] = 2.0*lossfac;
		}

		if (z == 0) {
		    xtrem++;
		    coeffs[(idx << 4) + 5] = coeff2 + 2.0*lossfac;
		    coeffs[(idx << 4) + 13] = 2.0*lossfac;
		}
		else if (z == 1) {
		    coeffs[(idx << 4) + 5] = coeff1 + lossfac;
		    coeffs[(idx << 4) + 13] = lossfac;
		    coeffs[(idx << 4) + 6] = coeff1 + lossfac;
		    coeffs[(idx << 4) + 14] = lossfac;
		}
		else {
		    xtrem++;
		    coeffs[(idx << 4) + 6] = coeff2 + 2.0*lossfac;
		    coeffs[(idx << 4) + 14] = 2.0*lossfac;
		}

		if (xtrem == 0) {
		    coeffs[(idx << 4) + 7] = 1.0;
		    coeffs[(idx << 4) + 8] = 1.0;
		}
		else if (xtrem == 1) {
		    coeffs[(idx << 4) + 7] = CPSI_FACE;
		    coeffs[(idx << 4) + 8] = APSI_FACE;
		}
		else if (xtrem == 2) {
		    coeffs[(idx << 4) + 7] = CPSI_EDGE;
		    coeffs[(idx << 4) + 8] = APSI_EDGE;
		}
		else {
		    coeffs[(idx << 4) + 7] = CPSI_CORNER;
		    coeffs[(idx << 4) + 8] = APSI_CORNER;
		}
		coeffs[(idx << 4) + 7] -= 6.0 * lossfac;

		idx++;
	    }
	}
    }

    nextCoeff = 27;
    drumShellCoeffs = -1;

#ifdef USE_GPU
    gpuAirboxIndexed = NULL;
#endif
}

AirboxIndexed::AirboxIndexed(string name)
    : Airbox(name)
{
}

AirboxIndexed::~AirboxIndexed()
{
    delete[] coeffs;
    delete[] index;

    if (energy) {
	delete[] energy;
	delete[] energyMask;
    }

#ifdef USE_GPU
    if (gpuAirboxIndexed) {
	delete gpuAirboxIndexed;
	u = NULL;
	u1 = NULL;
	u2 = NULL;
    }
#endif
}

void AirboxIndexed::swapBuffers(int n)
{
    if (reflection) {
	double *Psi = &(this->u[nxny]);
	int x, y, z;
	int idx = 0;
	for (z = 0; z < nz; z++) {
	    for (x = 0; x < nx; x++) {
		for (y = 0; y < ny; y++) {
		    if ((x == 0) || (x == (nx-1)) || (y == 0) || (y == (ny-1)) || (z == 0) || (z == (nz-1))) {
			Psi[idx] = 0.0;
		    }
		    idx++;
		}
	    }
	}
    }

    // we're doing the energy check in here so that it happens at the very end of
    // a timestep like it's supposed to. If we do it in runTimestep, the updates
    // from the embedded plates are potentially not done yet which could cause problems.
    if (energy) {
	// don't forget the halo!
	double *Psi = &(this->u1[nxny]);
	double *Psi1 = &(this->u2[nxny]);
	int x, y, z;
	int idx = 0;
	double svp;
	double kinpsi = 0.0, potpsi = 0.0;
	double k = GlobalSettings::getInstance()->getK();
	for (z = 0; z < nz; z++) {
	    for (x = 0; x < nx; x++) {
		for (y = 0; y < ny; y++) {
		    svp = 1.0;
		    if ((z == 0) || (z == (nz-1))) svp *= 0.5;
		    if ((y == 0) || (y == (ny-1))) svp *= 0.5;
		    if ((x == 0) || (x == (nx-1))) svp *= 0.5;

		    double psivel = (Psi[idx] - Psi1[idx]) / k;
		    kinpsi += 0.5 * (rho_a / (c_a*c_a)) * svp * psivel * psivel;
		    if (energyMask[idx] & 1) {
			// X
			potpsi += 0.5*rho_a*((Psi[idx+ny]-Psi[idx])*(Psi1[idx+ny]-Psi1[idx]))/(Q*Q);
		    }
		    if (energyMask[idx] & 2) {
			// Y
			potpsi += 0.5*rho_a*((Psi[idx+1]-Psi[idx])*(Psi1[idx+1]-Psi1[idx]))/(Q*Q);
		    }
		    if (energyMask[idx] & 4) {
			// Z
			potpsi += 0.5*rho_a*((Psi[idx+nxny]-Psi[idx])*(Psi1[idx+nxny]-Psi1[idx]))/(Q*Q);
		    }

		    idx++;
		}
	    }
	}
	energy[n] = Q*Q*Q*(kinpsi + potpsi);
    }

    Component::swapBuffers(n);
}


void AirboxIndexed::runTimestep(int n)
{
#ifdef USE_GPU
    if (gpuAirboxIndexed) {
	gpuAirboxIndexed->runTimestep(n, u, u1, u2);
	runInputs(n, &u[nxny], &u1[nxny], &u2[nxny]);
	return;
    }
#endif

    int i;
    double *c;

    // skip past halo
    double *Psi = &(this->u[nxny]);
    double *Psi0 = &(this->u2[nxny]);
    double *Psi1 = &(this->u1[nxny]);

    for (i = 0; i < ss; i++) {
	c = &coeffs[index[i] << 4];
	Psi[i] = c[8] * ((Psi1[i]*c[0] + Psi1[i+ny]*c[1] + Psi1[i-ny]*c[2] +
			  Psi1[i+1]*c[3] + Psi1[i-1]*c[4] + Psi1[i+nxny]*c[5] +
			  Psi1[i-nxny]*c[6]) -
			 (Psi0[i]*c[7] + Psi0[i+ny]*c[9] + Psi0[i-ny]*c[10] +
			  Psi0[i+1]*c[11] + Psi0[i-1]*c[12] + Psi0[i+nxny]*c[13] +
			  Psi0[i-nxny]*c[14]));
    }

    runInputs(n, Psi, Psi1, Psi0);
}

void AirboxIndexed::addDrumShellInternal(double cx, double cy, int startz, int endz, double R)
{
    int *MaskCirc_Psi;
    int i, j, z;
    double gridx, gridy;
    int idx, io;
    int zbase;

    /*
     * First generate a 2D circular mask with points inside the
     * drum set to 1
     */
    MaskCirc_Psi = new int[nx * ny];
    gridx = (((double)-(nx-1)) / 2.0) * Q;
    for (i = 0; i < nx; i++) {
	gridy = (((double)-(ny-1)) / 2.0) * Q;
	for (j = 0; j < ny; j++) {
	    if ((((gridx-cx)*(gridx-cx))+((gridy-cy)*(gridy-cy))) < (R*R)) {
		/* inside the circle */
		MaskCirc_Psi[(i*ny)+j] = 1;
	    }
	    else {
		MaskCirc_Psi[(i*ny)+j] = 0;
	    }
	    gridy += Q;
	}
	gridx += Q;
    }

    /*
     * Now mark the transitions between 0 and 1 in X and Y directions.
     * This is where the shell goes
     */
    for (i = 1; i < (nx-1); i++) {
	for (j = 1; j < (ny-1); j++) {
	    idx = j + (i*ny);
	    io = MaskCirc_Psi[idx] & 1;
	    if ((MaskCirc_Psi[idx+ny] & 1) != io) {
		MaskCirc_Psi[idx] |= 2;
	    }
	    if ((MaskCirc_Psi[idx-ny] & 1) != io) {
		MaskCirc_Psi[idx] |= 4;
	    }
	    if ((MaskCirc_Psi[idx+1] & 1) != io) {
		MaskCirc_Psi[idx] |= 8;
	    }
	    if ((MaskCirc_Psi[idx-1] & 1) != io) {
		MaskCirc_Psi[idx] |= 0x10;
	    }
	}
    }

    if (drumShellCoeffs < 0) {
	generateDrumShellCoefficients();
    }

    /*
     * Loop over Z layers that have the shell
     */
    for (z = startz; z <= endz; z++) {
	zbase = z * nx * ny;
	for (i = 1; i < (nx-1); i++) {
	    for (j = 1; j < (ny-1); j++) {
		idx = j + (i*ny);

		// modify to use correct co-efficient set
		switch (MaskCirc_Psi[idx] & 0x1e) {
		case 0x02: // XP
		    index[zbase + idx] = drumShellCoeffs + 0;
		    break;
		case 0x04: // XM
		    index[zbase + idx] = drumShellCoeffs + 1;
		    break;
		case 0x08: // YP
		    index[zbase + idx] = drumShellCoeffs + 2;
		    break;
		case 0x10: // YM
		    index[zbase + idx] = drumShellCoeffs + 3;
		    break;
		case 0x0a: // XPYP
		    index[zbase + idx] = drumShellCoeffs + 4;
		    break;
		case 0x12: // XPYM
		    index[zbase + idx] = drumShellCoeffs + 5;
		    break;
		case 0x0c: // XMYP
		    index[zbase + idx] = drumShellCoeffs + 6;
		    break;
		case 0x14: // XMYM
		    index[zbase + idx] = drumShellCoeffs + 7;
		    break;
		}

		// update energy mask as well
		if (energy) {
		    if (MaskCirc_Psi[idx] & 2) {
			// XP
			energyMask[zbase + idx] &= 0xfe;
		    }
		    if (MaskCirc_Psi[idx] & 8) {
			// YP
			energyMask[zbase + idx] &= 0xfd;
		    }
		}
	    }
	}
    }

    delete[] MaskCirc_Psi;
}

void AirboxIndexed::addDrumShell(double cx, double cy, double bz, double R, double H_shell)
{
    // work out inclusize start and end in Z dimension
    int NPHshell = (int)floor(H_shell / Q);

    int pd1 = floor((H_shell + bz) / Q);
    int pu1 = pd1 + 1;
    int pd2 = pd1 - NPHshell;
    int pu2 = pd2 + 1;

    int startz = pu2 - 1;
    int endz = pd1 - 1;

    addDrumShellInternal(cx, cy, startz, endz, R);
}

void AirboxIndexed::generateDrumShellCoefficients()
{
    /* Allocate 8 co-efficient sets */
    drumShellCoeffs = nextCoeff;
    nextCoeff += 8;

    int d = drumShellCoeffs;

    /*
     * Generate the 8 new co-efficient sets we need
     * Start by copying centre point co-efficients
     */
    memcpy(&coeffs[(d+0) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+1) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+2) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+3) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+4) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+5) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+6) << 4], &coeffs[13 << 4], 16 * sizeof(double));
    memcpy(&coeffs[(d+7) << 4], &coeffs[13 << 4], 16 * sizeof(double));

    double bdiff = (Gamma*Gamma) + (th*c_a*k/(Q*Q));
    double cdiff = th*c_a*k/(Q*Q);

    /* XP */
    coeffs[((d+0) << 4) + 0] += bdiff;
    coeffs[((d+0) << 4) + 1] -= bdiff;
    coeffs[((d+0) << 4) + 7] += cdiff;
    coeffs[((d+0) << 4) + 9] -= cdiff;
    /* XM */
    coeffs[((d+1) << 4) + 0] += bdiff;
    coeffs[((d+1) << 4) + 2] -= bdiff;
    coeffs[((d+1) << 4) + 7] += cdiff;
    coeffs[((d+1) << 4) + 10] -= cdiff;
    /* YP */
    coeffs[((d+2) << 4) + 0] += bdiff;
    coeffs[((d+2) << 4) + 3] -= bdiff;
    coeffs[((d+2) << 4) + 7] += cdiff;
    coeffs[((d+2) << 4) + 11] -= cdiff;
    /* YM */
    coeffs[((d+3) << 4) + 0] += bdiff;
    coeffs[((d+3) << 4) + 4] -= bdiff;
    coeffs[((d+3) << 4) + 7] += cdiff;
    coeffs[((d+3) << 4) + 12] -= cdiff;
    /* XPYP */
    coeffs[((d+4) << 4) + 0] += 2.0*bdiff;
    coeffs[((d+4) << 4) + 1] -= bdiff;
    coeffs[((d+4) << 4) + 3] -= bdiff;
    coeffs[((d+4) << 4) + 7] += 2.0*cdiff;
    coeffs[((d+4) << 4) + 9] -= cdiff;
    coeffs[((d+4) << 4) + 11] -= cdiff;
    /* XPYM */
    coeffs[((d+5) << 4) + 0] += 2.0*bdiff;
    coeffs[((d+5) << 4) + 1] -= bdiff;
    coeffs[((d+5) << 4) + 4] -= bdiff;
    coeffs[((d+5) << 4) + 7] += 2.0*cdiff;
    coeffs[((d+5) << 4) + 9] -= cdiff;
    coeffs[((d+5) << 4) + 12] -= cdiff;
    /* XMYP */
    coeffs[((d+6) << 4) + 0] += 2.0*bdiff;
    coeffs[((d+6) << 4) + 2] -= bdiff;
    coeffs[((d+6) << 4) + 3] -= bdiff;
    coeffs[((d+6) << 4) + 7] += 2.0*cdiff;
    coeffs[((d+6) << 4) + 10] -= cdiff;
    coeffs[((d+6) << 4) + 11] -= cdiff;
    /* XMYM */
    coeffs[((d+7) << 4) + 0] += 2.0*bdiff;
    coeffs[((d+7) << 4) + 2] -= bdiff;
    coeffs[((d+7) << 4) + 4] -= bdiff;
    coeffs[((d+7) << 4) + 7] += 2.0*cdiff;
    coeffs[((d+7) << 4) + 10] -= cdiff;
    coeffs[((d+7) << 4) + 12] -= cdiff;
}

/*
 * Adds a cylindrical drum shell, centred in the X and Y dimensions
 *
 * startz and endz are the inclusive z layers that have the shell
 */
void AirboxIndexed::addDrumShell(int startz, int endz, double R)
{
    addDrumShellInternal(0.0, 0.0, startz, endz, R);
}

int AirboxIndexed::getGPUMemRequired()
{
    return (ss * 3 * sizeof(double)) +
	(256 * sizeof(double)) + // coeffs
	(ss); // index
}

void AirboxIndexed::runPartialUpdate(int start, int len)
{
#ifdef USE_GPU
    if (gpuAirboxIndexed) {
	gpuAirboxIndexed->runPartialUpdate(u, u1, u2, start, len);
	return;
    }
#endif

    int i;
    double *c;

    // skip past halo
    double *Psi = &(this->u[nxny]);
    double *Psi0 = &(this->u2[nxny]);
    double *Psi1 = &(this->u1[nxny]);

    int end = (start + len) * nxny;

    for (i = (start * nxny); i < end; i++) {
	c = &coeffs[index[i] << 4];
	Psi[i] = c[8] * ((Psi1[i]*c[0] + Psi1[i+ny]*c[1] + Psi1[i-ny]*c[2] +
			  Psi1[i+1]*c[3] + Psi1[i-1]*c[4] + Psi1[i+nxny]*c[5] +
			  Psi1[i-nxny]*c[6]) -
			 (Psi0[i]*c[7] + Psi0[i+ny]*c[9] + Psi0[i-ny]*c[10] +
			  Psi0[i+1]*c[11] + Psi0[i-1]*c[12] + Psi0[i+nxny]*c[13] +
			  Psi0[i-nxny]*c[14]));
    }
}

bool AirboxIndexed::isOnGPU()
{
#ifdef USE_GPU
    if (gpuAirboxIndexed) return true;
#endif
    return false;
}

bool AirboxIndexed::moveToGPU()
{
#ifdef USE_GPU
    double *d_u, *d_u1, *d_u2;
    int i;
    gpuAirboxIndexed = new GPUAirboxIndexed(nx, ny, nz, &d_u, &d_u1, &d_u2,
					    index, coeffs);
    if (gpuAirboxIndexed->isOK()) {
	// move inputs to GPU as well
	for (i = 0; i < inputs.size(); i++) {
	    inputs[i]->moveToGPU();
	}
	delete[] u;
	delete[] u1;
	delete[] u2;
	u = d_u;
	u1 = d_u1;
	u2 = d_u2;
	logMessage(1, "Airbox %s moved to GPU", name.c_str());
	return true;
    }
    delete gpuAirboxIndexed;
    gpuAirboxIndexed = NULL;
    logMessage(1, "Unable to move airbox %s to GPU", name.c_str());
#endif
    return false;
}

double *AirboxIndexed::getEnergy()
{
    return energy;
}

/*
 * Ideally, the Airbox shouldn't have to know about the plates and that should
 * all be handled by the Embedding. For the most part it is, but when the energy
 * check is enabled, we need to know so that we don't do energy checks in the Z
 * dimension across a plate.
 *
 * zb is the Z layer of the airbox that is immediately below the plate.
 * true_Psi is a mask the size of an airbox layer which is set at points covered
 * by the plate and clear in other places.
 */
void AirboxIndexed::addPlate(int zb, double *true_Psi)
{
    int i;
    if (energy) {
	int zbase = zb * nxny;
	for (i = 0; i < nxny; i++) {
	    if (true_Psi[i] > 0.5) {
		// clear the Z flag for points directly under the plate
		energyMask[zbase+i] &= 0xfb;
	    }
	}
    }
}
