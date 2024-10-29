/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "SoundBoard.h"
#include "GlobalSettings.h"
#include "Logger.h"

extern "C" {
#include "pcg.h"
};

#include <cstdio>
#include <cstdlib>
using namespace std;

SoundBoard::SoundBoard(string name, double nu, double rho, double E, double thickness,
		       double tension, double lx, double ly, double t60_0, double t60_1000,
		       int bc, vector<ComponentString*> *strings)
    : Plate(name, nu, rho, E, thickness, tension, lx, ly, t60_0, t60_1000, bc)
{
    int i, j;
    CSRmatrix *tmp1, *tmp2, *tmp3;
    const double eps = 2.220446049250313e-16;
    double k = GlobalSettings::getInstance()->getK();

    this->strings = strings;

    // work out total string state size
    int sss = 0;
    for (i = 0; i < strings->size(); i++) {
	sss += (*strings)[i]->getStateSize();
    }
    logMessage(1, "Total string state size: %d", sss);

    // allocate matrices
    int nstr2 = 2 * strings->size();

    CSRmatrix *Is = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Is, sss, nstr2, 2 * strings->size());

    CSRmatrix *Js = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Js, sss, nstr2, 2 * strings->size());

    CSRmatrix *Ip = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Ip, ss, nstr2, 8 * strings->size());

    Jp = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Jp, ss, nstr2, 8 * strings->size());

    int start = 0;
    for (i = 0; i < strings->size(); i++) {
	ComponentString *str = (*strings)[i];

	// fill in Is
	CSRSetValue(Is, start, (i*2), 1.0);
	CSRSetValue(Is, start + str->getStateSize()-1, (i*2)+1, 1.0);

	// fill in Js
	double jfac = ((k*k) / (str->getRho() * (1.0 + (str->getSig0()*k)) * str->getH()));
	CSRSetValue(Js, start, (i*2), jfac);
	CSRSetValue(Js, start + str->getStateSize()-1, (i*2)+1, jfac);

	// fill in Ip
	double xc1 = str->getXc1();
	double xc2 = str->getXc2();
	double yc1 = str->getYc1();
	double yc2 = str->getYc2();

	double dnx = (double)(nx-1);
	double dny = (double)(ny-1);

	xc1 = 0.5 * (1.0 + fabs(xc1 - (1.0 / dnx)) - fabs(xc1 - ((dnx - 1.0) / dnx)));
	xc2 = 0.5 * (1.0 + fabs(xc2 - (1.0 / dnx)) - fabs(xc2 - ((dnx - 1.0) / dnx)));
	yc1 = 0.5 * (1.0 + fabs(yc1 - (1.0 / dny)) - fabs(yc1 - ((dny - 1.0) / dny)));
	yc2 = 0.5 * (1.0 + fabs(yc2 - (1.0 / dny)) - fabs(yc2 - ((dny - 1.0) / dny)));

	//logMessage(1, "%f, %f, %f, %f", xc1, xc2, yc1, yc2);

	int xc1_int = (int)floor(xc1 * dnx * (1.0 - eps));
	int yc1_int = (int)floor(yc1 * dny * (1.0 - eps));
	int xc2_int = (int)floor(xc2 * dnx * (1.0 - eps));
	int yc2_int = (int)floor(yc2 * dny * (1.0 - eps));

	double xc1_frac = (xc1 * dnx) - ((double)xc1_int);
	double yc1_frac = (yc1 * dny) - ((double)yc1_int);
	double xc2_frac = (xc2 * dnx) - ((double)xc2_int);
	double yc2_frac = (yc2 * dny) - ((double)yc2_int);

	int ind1 = (xc1_int)*ny + yc1_int;
	int ind2 = (xc2_int)*ny + yc2_int;

	CSRSetValue(Ip, ind1, (i*2), (1.0 - xc1_frac) * (1.0 - yc1_frac));
	CSRSetValue(Ip, ind1+1, (i*2), (1.0 - xc1_frac) * (yc1_frac));
	CSRSetValue(Ip, ind1+ny, (i*2), (xc1_frac) * (1.0 - yc1_frac));
	CSRSetValue(Ip, ind1+ny+1, (i*2), (xc1_frac) * (yc1_frac));

	CSRSetValue(Ip, ind2, (i*2)+1, (1.0 - xc2_frac) * (1.0 - yc2_frac));
	CSRSetValue(Ip, ind2+1, (i*2)+1, (1.0 - xc2_frac) * (yc2_frac));
	CSRSetValue(Ip, ind2+ny, (i*2)+1, (xc2_frac) * (1.0 - yc2_frac));
	CSRSetValue(Ip, ind2+ny+1, (i*2)+1, (xc2_frac) * (yc2_frac));

	// fill in Jp
	jfac = ((k*k) / (rho * thickness * (1.0 + (sig_0 * k)) * h * h));
	CSRSetValue(Jp, ind1, (i*2), jfac * (1.0 - xc1_frac) * (1.0 - yc1_frac));
	CSRSetValue(Jp, ind1+1, (i*2), jfac * (1.0 - xc1_frac) * (yc1_frac));
	CSRSetValue(Jp, ind1+ny, (i*2), jfac * (xc1_frac) * (1.0 - yc1_frac));
	CSRSetValue(Jp, ind1+ny+1, (i*2), jfac * (xc1_frac) * (yc1_frac));

	CSRSetValue(Jp, ind2, (i*2)+1, jfac * (1.0 - xc2_frac) * (1.0 - yc2_frac));
	CSRSetValue(Jp, ind2+1, (i*2)+1, jfac * (1.0 - xc2_frac) * (yc2_frac));
	CSRSetValue(Jp, ind2+ny, (i*2)+1, jfac * (xc2_frac) * (1.0 - yc2_frac));
	CSRSetValue(Jp, ind2+ny+1, (i*2)+1, jfac * (xc2_frac) * (yc2_frac));

	start += str->getStateSize();
    }

    // compute Msp (Is'*Js + Ip'*Jp)
    tmp1 = CSR_transpose(Is);
    tmp2 = CSR_matrix_multiply(tmp1, Js);
    CSRmatrix *IpT = CSR_transpose(Ip);
    tmp3 = CSR_matrix_multiply(IpT, Jp);
    CSRmatrix *Msp = CSR_matrix_add(tmp2, tmp3);
    CSR_free(tmp1);
    CSR_free(tmp2);
    CSR_free(tmp3);

    // compute inverse of MsP
    double *mspinvdat = invertMatrix(Msp);

    // it's usually pretty sparse, so turn it into a sparse matrix
    // FIXME: may want to be on the lookout for this getting too dense, and store
    // it as a dense matrix if it does
    Mspinv = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Mspinv, nstr2, nstr2, nstr2*2);
    for (i = 0; i < nstr2; i++) {
	for (j = 0; j < nstr2; j++) {
	    double val = mspinvdat[(i*nstr2)+j];
	    if (fabs(val) > 1e-15) {
		CSRSetValue(Mspinv, i, j, val);
	    }
	}
    }
    free(mspinvdat);

    // allocate vectors for use in connection update
    l = new double[nstr2];
    fsp = new double[nstr2];

    // compute IpB and IpC
    IpB = CSR_matrix_multiply(IpT, B);
    IpC = CSR_matrix_multiply(IpT, C);
    CSR_free(IpT);

    // IsB, IsC, Js equivalents are calculated in the string constructor    

    CSR_free(Msp);

    CSR_free(Is);
    CSR_free(Ip);
    CSR_free(Js);
}

void SoundBoard::logMatrices()
{
    Plate::logMatrices();
    saveMatrix(IpB, "IpB");
    saveMatrix(IpC, "IpC");
    saveMatrix(Jp, "Jp");
    saveMatrix(Mspinv, "Mspinv");
}

SoundBoard::~SoundBoard()
{
    CSR_free(IpB);
    CSR_free(IpC);
    CSR_free(Jp);
    CSR_free(Mspinv);

    delete[] l;
    delete[] fsp;

    // actual strings will be deleted by the top level Instrument, but the
    // vector is ours
    delete strings;
}

void SoundBoard::runTimestep(int n)
{
    int i;

    Plate::runTimestep(n);

    // now do connections with strings
    // assemble vector l
    // get values from soundboard
    CSR_matrix_vector_mult(IpB, u1, l);
    CSR_matrix_vector(IpC, u2, l, FALSE, OP_ADD);

    // now get values from strings
    for (i = 0; i < strings->size(); i++) {
	ComponentString *str = (*strings)[i];
	str->getConnectionValues(&l[i*2]);
    }
    
    // multiply l by Mspinv
    CSR_matrix_vector_mult(Mspinv, l, fsp);

    // update soundboard based on result
    CSR_matrix_vector(Jp, fsp, u, FALSE, OP_SUB);

    // update strings based on result
    for (i = 0; i < strings->size(); i++) {
	ComponentString *str = (*strings)[i];
	str->setConnectionValues(&fsp[i*2]);
    }
}

int SoundBoard::getGPUScore()
{
    return GPU_SCORE_NO;
}

bool SoundBoard::isThreadable()
{
    return false;
}
