/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */
/*
 * This code "unrolls" a CSRmatrix structure into sets of stencil co-efficients that can be
 * applied by a looping code.
 *
 * Constraints:
 *  - the co-efficients for each location must be the same, except for at the boundaries where
 *    different co-efficients are allowed for the edges, corners, and one element in from them
 *  - all co-efficients for each point must fit within a 13 point diamond-shaped stencil
 *
 * The "unrollMatrixWithIndex" function has less constraints. The stencil must still fit
 * within the 13 point diamond, but there can be up to 256 unique sets of co-efficients in the
 * matrix, and each one may occur anywhere.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "csrmatrix.h"

/* types of co-efficient (in one dimension) */
enum {
    TYPE_0,          /* far left/top edge */
    TYPE_1,          /* one element in from left/top edge */
    TYPE_MIDDLE,     /* middle element */
    TYPE_N_MINUS_1,  /* one element in from right/bottom edge*/
    TYPE_N,          /* far right/bottom edge */
    TYPE_MAX
};

/* largest stencil size is a diamond with 13 co-efficients */
#define MAX_COEFFS 13

typedef struct coeffSet_s
{
    int valid;
    int numCoeffs;
    int offsets[MAX_COEFFS];   /* relative to the row number */
    double coeffs[MAX_COEFFS];
} coeffSet_t;

static coeffSet_t coeffSets[256];

static void printCoeffSet(coeffSet_t *c, int type, int nx, int ny)
{
    int i, xoffs, yoffs, o;
    printf("X type %d, Y type %d, contains %d coeffs:\n", type/TYPE_MAX, type%TYPE_MAX, c->numCoeffs);
    for (i = 0; i < c->numCoeffs; i++) {
	xoffs = 0;
	yoffs = 0;
	o = c->offsets[i];
	while (o < -2) {
	    xoffs--;
	    o += (ny+1);
	}
	while (o > 2) {
	    xoffs++;
	    o -= (ny+1);
	}
	yoffs = o;

	printf(" - offset %d (%d,%d), value %f\n", c->offsets[i], xoffs, yoffs, c->coeffs[i]);
    }
}

/*
 * Indices of the stencil elements within each co-efficient set
 */
#define STENCIL_FARLEFT     0
#define STENCIL_UP_LEFT     1
#define STENCIL_LEFT        2
#define STENCIL_DOWN_LEFT   3
#define STENCIL_TOP         4
#define STENCIL_UP          5
#define STENCIL_MIDDLE      6
#define STENCIL_DOWN        7
#define STENCIL_BOTTOM      8
#define STENCIL_UP_RIGHT    9
#define STENCIL_RIGHT      10
#define STENCIL_DOWN_RIGHT 11
#define STENCIL_FARRIGHT   12

/*
 * Puts the co-efficients extracted from the matrix into the array in the correct locations
 */
static void convertCoeffSet(coeffSet_t *c, int type, int nx, int ny, double *coeffs)
{
    int i, xoffs, yoffs, o;
    for (i = 0; i < c->numCoeffs; i++) {
	/* convert the flat offset to a 2D offset */
	xoffs = 0;
	yoffs = 0;
	o = c->offsets[i];
	while (o < -2) {
	    xoffs--;
	    o += (ny+1);
	}
	while (o > 2) {
	    xoffs++;
	    o -= (ny+1);
	}
	yoffs = o;

	/* store this value according to the offset */
	switch (xoffs) {
	case -2:
	    coeffs[STENCIL_FARLEFT] = c->coeffs[i];
	    break;
	case -1:
	    switch (yoffs) {
	    case -1:
		coeffs[STENCIL_UP_LEFT] = c->coeffs[i];
		break;
	    case 0:
		coeffs[STENCIL_LEFT] = c->coeffs[i];
		break;
	    case 1:
		coeffs[STENCIL_DOWN_LEFT] = c->coeffs[i];
		break;
	    }
	    break;
	case 0:
	    switch (yoffs) {
	    case -2:
		coeffs[STENCIL_TOP] = c->coeffs[i];
		break;
	    case -1:
		coeffs[STENCIL_UP] = c->coeffs[i];
		break;
	    case 0:
		coeffs[STENCIL_MIDDLE] = c->coeffs[i];
		break;
	    case 1:
		coeffs[STENCIL_DOWN] = c->coeffs[i];
		break;
	    case 2:
		coeffs[STENCIL_BOTTOM] = c->coeffs[i];
		break;
	    }
	    break;
	case 1:
	    switch (yoffs) {
	    case -1:
		coeffs[STENCIL_UP_RIGHT] = c->coeffs[i];
		break;
	    case 0:
		coeffs[STENCIL_RIGHT] = c->coeffs[i];
		break;
	    case 1:
		coeffs[STENCIL_DOWN_RIGHT] = c->coeffs[i];
		break;
	    }
	    break;
	case 2:
	    coeffs[STENCIL_FARRIGHT] = c->coeffs[i];
	    break;
	}
    }
}

/*
 * Compares two doubles for equality, within the tolerance
 */
#define TOLERANCE 1e-15

static int doubleCompare(double v1, double v2)
{
    if (fabs(v1 - v2) < TOLERANCE) return 1;
    return 0;
}

/*
 * coeffs should be an array of 325 doubles to receive the 25 sets of 13 co-efficients
 * Returns 1 on success, 0 on failure
 */
int unrollMatrix(CSRmatrix *m, int nx, int ny, double *coeffs)
{
    int ss = ((nx+1)*(ny+1));
    int i, j;
    int x, y;
    int xtype, ytype, type;
    int ncoeffs;
    int ncoeffsets;

    /* zero the co-efficients by default */
    memset(&coeffSets, 0, sizeof(coeffSet_t)*TYPE_MAX*TYPE_MAX);
    memset(coeffs, 0, 325 * sizeof(double));

    /*printf("Problem size is %dx%d\n", nx, ny);
      printf("Matrix size is %dx%d\n", m->ncol, m->nrow);*/
    if (m->ncol != ss) {
	printf("ERROR: matrix width not equal to (nx+1)*(ny+1), cannot unroll matrix\n");
	return 0;
    }
    if (m->nrow != ss) {
	printf("ERROR: matrix height not equal to (nx+1)*(ny+1), cannot unroll matrix\n");
	return 0;
    }

    x = 0;
    y = 0;
    for (i = 0; i < ss; i++) {
	ncoeffs = m->rowStart[i+1] - m->rowStart[i];
	if (ncoeffs > MAX_COEFFS) {
	    printf("ERROR: more than %d coeffs in row %d, cannot unroll matrix\n", MAX_COEFFS, i);
	    return 0;
	}

	/* work out which "type" of element this is (are we close to a boundary?) */
	xtype = TYPE_MIDDLE;
	if (x == 0) {
	    xtype = TYPE_0;
	}
	else if (x == 1) {
	    xtype = TYPE_1;
	}
	else if (x == (nx-1)) {
	    xtype = TYPE_N_MINUS_1;
	}
	else if (x == nx) {
	    xtype = TYPE_N;
	}

	ytype = TYPE_MIDDLE;
	if (y == 0) {
	    ytype = TYPE_0;
	}
	else if (y == 1) {
	    ytype = TYPE_1;
	}
	else if (y == (ny-1)) {
	    ytype = TYPE_N_MINUS_1;
	}
	else if (y == ny) {
	    ytype = TYPE_N;
	}

	type = (xtype * TYPE_MAX) + ytype;

	/* see if we found this "type" already */
	if (coeffSets[type].valid) {
	    /* compare to existing one. must match */
	    if (coeffSets[type].numCoeffs != ncoeffs) {
		printf("ERROR: conflicting numbers of coefficients for type %d! cannot unroll matrix\n", type);
		return 0;
	    }
	    else {
		for (j = 0; j < ncoeffs; j++) {
		    if (coeffSets[type].offsets[j] != (m->colIndex[m->rowStart[i] + j] - i)) {
			printf("ERROR: conflicting offsets for type %d! cannot unroll matrix\n", type);
			return 0;
		    }
		    if (!doubleCompare(coeffSets[type].coeffs[j], m->values[m->rowStart[i] + j])) {
			printf("ERROR: conflicting coeff values for type %d! cannot unroll matrix\n", type);
			return 0;
		    }
		}
	    }
	}
	else {
	    /* store newly discovered co-efficients */
	    coeffSets[type].valid = 1;
	    coeffSets[type].numCoeffs = ncoeffs;
	    for (j = 0; j < ncoeffs; j++) {
		coeffSets[type].offsets[j] = m->colIndex[m->rowStart[i] + j] - i;
		coeffSets[type].coeffs[j] = m->values[m->rowStart[i] + j];
	    }
	}

	y++;
	if (y >= (ny+1)) {
	    y = 0;
	    x++;
	}
    }

    /*
     * Now we should have all the co-efficients. Store them in the passed array
     */
    ncoeffsets = 0;
    for (i = 0; i < (TYPE_MAX*TYPE_MAX); i++) {
	if (coeffSets[i].valid) {
	    convertCoeffSet(&coeffSets[i], i, nx, ny, &coeffs[i*13]);
	    ncoeffsets++;
	}
    }
    return 1;
}

/*
 * This function unrolls a sparse matrix that may have up to 256 distinct sets of
 * co-efficients. (More than 256 will trigger an error as it cannot be indexed by a char).
 *
 * Allocates coefficient set storage dynamically into coeffs, and returns number of them in
 * numCoeffSets. Each one is 13 elements. index array is assumed to be already allocated and
 * of size (nx+1)*(ny+1) and is filled in to point to right coefficient set for each point.
 */
int unrollMatrixWithIndex(CSRmatrix *m, int nx, int ny, int *numCoeffSets, double **coeffs, unsigned char *index)
{
    int ss = ((nx+1)*(ny+1));
    int i, j, k;
    int ncoeffs;
    int ncoeffsets;
    int match;
    
    /* zero the co-efficients by default */
    memset(&coeffSets, 0, sizeof(coeffSet_t)*256);
    
    if (m->ncol != ss) {
	printf("ERROR: matrix width not equal to (nx+1)*(ny+1), cannot unroll matrix\n");
	return 0;
    }
    if (m->nrow != ss) {
	printf("ERROR: matrix height not equal to (nx+1)*(ny+1), cannot unroll matrix\n");
	return 0;
    }

    ncoeffsets = 0;
    /* loop over each matrix row */
    for (i = 0; i < ss; i++) {
	ncoeffs = m->rowStart[i+1] - m->rowStart[i];
	if (ncoeffs > MAX_COEFFS) {
	    printf("ERROR: more than %d coeffs in row %d, cannot unroll matrix\n", MAX_COEFFS, i);
	    return 0;
	}

	match = 0;
	/* check if this row matches any of the existing co-efficient sets */
	for (j = 0; j < ncoeffsets; j++) {
	    match = 0;
	    /* test correct size first, for speed */
	    if (coeffSets[j].numCoeffs == ncoeffs) {
		match = 1;
		/* now check values and indices match */
		for (k = 0; k < ncoeffs; k++) {
		    if (coeffSets[j].offsets[k] != (m->colIndex[m->rowStart[i] + k] - i)) {
			match = 0;
			break;
		    }
		    if (!doubleCompare(coeffSets[j].coeffs[k], m->values[m->rowStart[i] + k])) {
			match = 0;
			break;
		    }
		}
		if (match) break;
	    }
	}

	if (!match) {
	    /* no match, create a new one */
	    j = ncoeffsets;
	    ncoeffsets++;
	    if (ncoeffsets > 256) {
		printf("ERROR: more than 256 coeff sets in matrix, cannot unroll\n");
		return 0;
	    }

	    coeffSets[j].valid = 1;
	    coeffSets[j].numCoeffs = ncoeffs;
	    for (k = 0; k < ncoeffs; k++) {
		coeffSets[j].offsets[k] = m->colIndex[m->rowStart[i] + k] - i;
		coeffSets[j].coeffs[k] = m->values[m->rowStart[i] + k];
	    }
	}

	/* fill in index array */
	index[i] = j;
    }

    /* allocate coefficient return array */
    *coeffs = (double *)malloc(13 * ncoeffsets * sizeof(double));
    memset(*coeffs, 0, sizeof(double)*ncoeffsets*13);
    *numCoeffSets = ncoeffsets;

    /* copy to return array */
    for (i = 0; i < ncoeffsets; i++) {
	convertCoeffSet(&coeffSets[i], i, nx, ny, &((*coeffs)[i*13]));
    }
    return 1;
}
