/*
 * Copyright (c) 2012-2015 The University of Edinburgh 
 */
#ifndef _MATGEN_H_
#define _MATGEN_H_

#include "csrmatrix.h"

/*
 * This implements both the Matlab matrix generation functions in one.
 *
 * D, scalematx and scalematy are optional and will not be generated if the pointers are
 * NULL.
 *
 * Although scalematx and scalematy are matrices, they are dense, so returning as a simple
 * array of doubles is more efficient than using a CSRmatrix. Size is (Nx, Ny+1) for scalematx
 * and (Nx+1, Ny) for scalematy.
 */
void plateMatGen(int nx, int ny, int bc, double nu, CSRmatrix **DD, CSRmatrix **D,
		 CSRmatrix **D2, CSRmatrix **D3, CSRmatrix **Dxy2, double **scalevec,
		 double **scalematx, double **scalematy, CSRmatrix **Dx, CSRmatrix **Dy);

#endif
