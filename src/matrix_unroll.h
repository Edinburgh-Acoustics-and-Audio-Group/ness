/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
#ifndef _MATRIX_UNROLL_H_
#define _MATRIX_UNROLL_H_

#include "csrmatrix.h"

int unrollMatrixWithIndex(CSRmatrix *m, int nx, int ny, int *numCoeffSets, double **coeffs, unsigned char *index);

#endif
