/*
 * Copyright (c) 2012-2015 The University of Edinburgh 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matgen.h"


/*
 * Multiplies row row of matrix in1 by column row of matrix in2
 * Assumes that vals and colindex can hold at least 13 values
 * Column indices are adjusted by row number
 * Returns number of elements in row, -1 on error
 */
static int multiplySingleRow(CSRmatrix *in1, CSRmatrix *in2, int row, double *vals, int *colindex)
{
    int i, j, k, col;
    int idx = 0;
    double sum;

    /* loop across columns of result */
    for (i = 0; i < in2->ncol; i++) {
	sum = 0.0;
	/* loop across row of matrix 1 */
	for (j = in1->rowStart[row]; j < in1->rowStart[row+1]; j++) {
	    col = in1->colIndex[j];
	    /* find corresponding element in matrix 2 if any */
	    for (k = in2->rowStart[col]; k < in2->rowStart[col+1]; k++) {
		if (in2->colIndex[k] == i) {
		    sum = sum + (in1->values[j] * in2->values[k]);
		    break;
		}
		if (in2->colIndex[k] > i) break;
	    }
	}
	if (sum != 0.0) {
	    if (idx >= 13) return -1;
	    vals[idx] = sum;
	    colindex[idx] = i - row;
	    idx++;
	}
    }

    return idx;
}

/*
 * This is an optimised version of the code that computes DD. The formula is:
 *
 *  DD = Dxx1*D2 + Dyy1*D3 + scaler*Dxy1*Dxy2
 *
 * However the three matrix multiplications are very slow for large problem sizes. Because
 * the final matrix contains a lot of repetition and only at most 25 unique rows, we can
 * do it much faster by only computing 25 key rows and then replicating them over the rest
 * of the result matrix.
 *
 * With the optimisation, computing DD for a 200x200 plate is near-instantaneous. Without,
 * it took approximately 2 minutes in testing.
 */
static CSRmatrix *computeDD(CSRmatrix *Dxx1, CSRmatrix *D2, CSRmatrix *Dyy1, CSRmatrix *D3, CSRmatrix *Dxy1, CSRmatrix *Dxy2, double scaler, int nx, int ny)
{
    static double vals[13 * 25 * 3];
    static int colindex[13 * 25 * 3];
    int n;
    static double finalvals[13 * 25];
    int finalcols[13];
    int i, j, idx, k;
    int x, y;

    CSRmatrix *DD;

    memset(colindex, 0, 13*25*3*sizeof(int));
    memset(vals, 0, 13*25*3*sizeof(double));

    /* first get all the rows we need */
    n = multiplySingleRow(Dxx1, D2, 0, &vals[13*0], &colindex[13*0]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, 1, &vals[13*1], &colindex[13*1]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, 2, &vals[13*2], &colindex[13*2]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny-1, &vals[13*3], &colindex[13*3]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny, &vals[13*4], &colindex[13*4]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+1, &vals[13*5], &colindex[13*5]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+2, &vals[13*6], &colindex[13*6]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+3, &vals[13*7], &colindex[13*7]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+ny, &vals[13*8], &colindex[13*8]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+ny+1, &vals[13*9], &colindex[13*9]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+ny+2, &vals[13*10], &colindex[13*10]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+ny+3, &vals[13*11], &colindex[13*11]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, ny+ny+4, &vals[13*12], &colindex[13*12]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, 3*ny+1, &vals[13*13], &colindex[13*13]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, 3*ny+2, &vals[13*14], &colindex[13*14]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*(nx-1), &vals[13*15], &colindex[13*15]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*(nx-1)+1, &vals[13*16], &colindex[13*16]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*(nx-1)+2, &vals[13*17], &colindex[13*17]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*nx-2, &vals[13*18], &colindex[13*18]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*nx-1, &vals[13*19], &colindex[13*19]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*nx, &vals[13*20], &colindex[13*20]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*nx+1, &vals[13*21], &colindex[13*21]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*nx+2, &vals[13*22], &colindex[13*22]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*(nx+1)-2, &vals[13*23], &colindex[13*23]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxx1, D2, (ny+1)*(nx+1)-1, &vals[13*24], &colindex[13*24]);
    if (n < 0) return NULL;


    n = multiplySingleRow(Dyy1, D3, 0, &vals[13*25], &colindex[13*25]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, 1, &vals[13*26], &colindex[13*26]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, 2, &vals[13*27], &colindex[13*27]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny-1, &vals[13*28], &colindex[13*28]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny, &vals[13*29], &colindex[13*29]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+1, &vals[13*30], &colindex[13*30]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+2, &vals[13*31], &colindex[13*31]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+3, &vals[13*32], &colindex[13*32]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+ny, &vals[13*33], &colindex[13*33]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+ny+1, &vals[13*34], &colindex[13*34]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+ny+2, &vals[13*35], &colindex[13*35]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+ny+3, &vals[13*36], &colindex[13*36]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, ny+ny+4, &vals[13*37], &colindex[13*37]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, 3*ny+1, &vals[13*38], &colindex[13*38]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, 3*ny+2, &vals[13*39], &colindex[13*39]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*(nx-1), &vals[13*40], &colindex[13*40]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*(nx-1)+1, &vals[13*41], &colindex[13*41]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*(nx-1)+2, &vals[13*42], &colindex[13*42]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*nx-2, &vals[13*43], &colindex[13*43]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*nx-1, &vals[13*44], &colindex[13*44]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*nx, &vals[13*45], &colindex[13*45]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*nx+1, &vals[13*46], &colindex[13*46]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*nx+2, &vals[13*47], &colindex[13*47]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*(nx+1)-2, &vals[13*48], &colindex[13*48]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dyy1, D3, (ny+1)*(nx+1)-1, &vals[13*49], &colindex[13*49]);
    if (n < 0) return NULL;


    n = multiplySingleRow(Dxy1, Dxy2, 0, &vals[13*50], &colindex[13*50]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, 1, &vals[13*51], &colindex[13*51]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, 2, &vals[13*52], &colindex[13*52]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny-1, &vals[13*53], &colindex[13*53]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny, &vals[13*54], &colindex[13*54]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+1, &vals[13*55], &colindex[13*55]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+2, &vals[13*56], &colindex[13*56]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+3, &vals[13*57], &colindex[13*57]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+ny, &vals[13*58], &colindex[13*58]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+ny+1, &vals[13*59], &colindex[13*59]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+ny+2, &vals[13*60], &colindex[13*60]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+ny+3, &vals[13*61], &colindex[13*61]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, ny+ny+4, &vals[13*62], &colindex[13*62]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, 3*ny+1, &vals[13*63], &colindex[13*63]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, 3*ny+2, &vals[13*64], &colindex[13*64]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*(nx-1), &vals[13*65], &colindex[13*65]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*(nx-1)+1, &vals[13*66], &colindex[13*66]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*(nx-1)+2, &vals[13*67], &colindex[13*67]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*nx-2, &vals[13*68], &colindex[13*68]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*nx-1, &vals[13*69], &colindex[13*69]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*nx, &vals[13*70], &colindex[13*70]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*nx+1, &vals[13*71], &colindex[13*71]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*nx+2, &vals[13*72], &colindex[13*72]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*(nx+1)-2, &vals[13*73], &colindex[13*73]);
    if (n < 0) return NULL;
    n = multiplySingleRow(Dxy1, Dxy2, (ny+1)*(nx+1)-1, &vals[13*74], &colindex[13*74]);
    if (n < 0) return NULL;

    memset(finalvals, 0, 13*25*sizeof(double));

    /* convert each row into final 13 values in known locations */
    for (i = 0; i < 25; i++) {
	for (j = 0; j < 13; j++) {
	    /* add contribution from Dxx1 and D2 */
	    if (colindex[i*13+j] == -(ny+ny+2)) {
		finalvals[i*13] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == -(ny+2)) {
		finalvals[i*13+1] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == -(ny+1)) {
		finalvals[i*13+2] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == -(ny)) {
		finalvals[i*13+3] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == -2) {
		finalvals[i*13+4] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == -1) {
		finalvals[i*13+5] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == 0) {
		finalvals[i*13+6] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == 1) {
		finalvals[i*13+7] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == 2) {
		finalvals[i*13+8] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == ny) {
		finalvals[i*13+9] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == (ny+1)) {
		finalvals[i*13+10] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == (ny+2)) {
		finalvals[i*13+11] += vals[i*13+j];
	    }
	    else if (colindex[i*13+j] == (ny+ny+2)) {
		finalvals[i*13+12] += vals[i*13+j];
	    }

	    /* add contribution from Dyy1 and D3 */
	    if (colindex[(i+25)*13+j] == -(ny+ny+2)) {
		finalvals[i*13] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == -(ny+2)) {
		finalvals[i*13+1] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == -(ny+1)) {
		finalvals[i*13+2] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == -(ny)) {
		finalvals[i*13+3] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == -2) {
		finalvals[i*13+4] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == -1) {
		finalvals[i*13+5] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == 0) {
		finalvals[i*13+6] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == 1) {
		finalvals[i*13+7] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == 2) {
		finalvals[i*13+8] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == ny) {
		finalvals[i*13+9] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == (ny+1)) {
		finalvals[i*13+10] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == (ny+2)) {
		finalvals[i*13+11] += vals[(i+25)*13+j];
	    }
	    else if (colindex[(i+25)*13+j] == (ny+ny+2)) {
		finalvals[i*13+12] += vals[(i+25)*13+j];
	    }

	    /* add contribution from Dxy1 and Dxy2 (scaled) */
	    if (colindex[(i+50)*13+j] == -(ny+ny+2)) {
		finalvals[i*13] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == -(ny+2)) {
		finalvals[i*13+1] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == -(ny+1)) {
		finalvals[i*13+2] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == -(ny)) {
		finalvals[i*13+3] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == -2) {
		finalvals[i*13+4] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == -1) {
		finalvals[i*13+5] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == 0) {
		finalvals[i*13+6] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == 1) {
		finalvals[i*13+7] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == 2) {
		finalvals[i*13+8] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == ny) {
		finalvals[i*13+9] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == (ny+1)) {
		finalvals[i*13+10] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == (ny+2)) {
		finalvals[i*13+11] += vals[(i+50)*13+j] * scaler;
	    }
	    else if (colindex[(i+50)*13+j] == (ny+ny+2)) {
		finalvals[i*13+12] += vals[(i+50)*13+j] * scaler;
	    }
	}
    }

    finalcols[0] = -(ny+ny+2);
    finalcols[1] = -(ny+2);
    finalcols[2] = -(ny+1);
    finalcols[3] = -ny;
    finalcols[4] = -2;
    finalcols[5] = -1;
    finalcols[6] = 0;
    finalcols[7] = 1;
    finalcols[8] = 2;
    finalcols[9] = ny;
    finalcols[10] = ny+1;
    finalcols[11] = ny+2;
    finalcols[12] = ny+ny+2;

    /* build DD! */
    DD = malloc(sizeof(CSRmatrix));
    if (!DD) return NULL;
    if (!CSR_setup(DD, ((nx+1)*(ny+1)), ((nx+1)*(ny+1)), ((nx+1)*(ny+1)*13))) {
	free(DD);
	return NULL;
    }

    /* loop over rows of DD */
    idx = 0;
    x = 0;
    y = 0;
    for (i = 0; i < ((nx+1)*(ny+1)); i++) {
	DD->rowStart[i] = idx;
	
	j = 12;
	if (x == 0) {
	    if (y == 0) j = 0;
	    else if (y == 1) j = 1;
	    else if (y == (ny-1)) j = 3;
	    else if (y == ny) j = 4;
	    else j = 2;
	}
	else if (x == 1) {
	    if (y == 0) j = 5;
	    else if (y == 1) j = 6;
	    else if (y == (ny-1)) j = 8;
	    else if (y == ny) j = 9;
	    else j = 7;

	}
	else if (x == (nx-1)) {
	    if (y == 0) j = 15;
	    else if (y == 1) j = 16;
	    else if (y == (ny-1)) j = 18;
	    else if (y == ny) j = 19;
	    else j = 17;

	}
	else if (x == nx) {
	    if (y == 0) j = 20;
	    else if (y == 1) j = 21;
	    else if (y == (ny-1)) j = 23;
	    else if (y == ny) j = 24;
	    else j = 22;

	}
	else {
	    if (y == 0) j = 10;
	    else if (y == 1) j = 11;
	    else if (y == (ny-1)) j = 13;
	    else if (y == ny) j = 14;
	}

	for (k = 0; k < 13; k++) {
	    if (finalvals[j*13 + k] != 0.0) {
		DD->colIndex[idx] = finalcols[k] + i;
		DD->values[idx] = finalvals[j*13 + k];
		idx++;
	    }
	}

	/* update row and column */
	y++;
	if (y == (ny+1)) {
	    y = 0;
	    x++;
	}
    }
    DD->rowStart[i] = idx;
   
    return DD;
}


void plateMatGen(int nx, int ny, int bc, double nu, CSRmatrix **DD, CSRmatrix **D, CSRmatrix **D2, CSRmatrix **D3, CSRmatrix **Dxy2, double **scalevec, double **scalematx, double **scalematy, CSRmatrix **Dx, CSRmatrix **Dy)
{
    CSRmatrix *sx0, *sx1, *sxnu, *sy1, *sy0, *synu;
    CSRmatrix *Dxxc, *Dyyc, *Dxx, *Dyy, *Dxx1, *Dyy1;
    CSRmatrix *sx2 = NULL, *sy3 = NULL, *sx31 = NULL, *sx32 = NULL, *sy21 = NULL, *sy22 = NULL;
    CSRmatrix *sx, *sy;
    CSRmatrix *Dx1, *Dy1, *Dxy1, *Dx2, *Dy2;
    CSRmatrix *tmp, *tmp2, *tmp3;

    double *diag1, *diag2, *diag3, *diag4;
    double *scalevecx, *scalevecy;

    int i, j;
    int ss = (nx+1) * (ny+1);

    sx1 = CSR_create_eye(nx+1);
    sx0 = CSR_create_eye(nx+1);
    CSRSetValue(sx0, 0, 0, 0.0);
    CSRSetValue(sx0, nx, nx, 0.0);
    sxnu = CSR_create_eye(nx+1);
    CSRSetValue(sxnu, 0, 0, (1.0-(nu*nu)));
    CSRSetValue(sxnu, nx, nx, (1.0-(nu*nu)));

    sy1 = CSR_create_eye(ny+1);
    sy0 = CSR_create_eye(ny+1);
    CSRSetValue(sy0, 0, 0, 0.0);
    CSRSetValue(sy0, ny, ny, 0.0);
    synu = CSR_create_eye(ny+1);
    CSRSetValue(synu, 0, 0, (1.0-(nu*nu)));
    CSRSetValue(synu, ny, ny, (1.0-(nu*nu)));

    diag1 = malloc((nx+1)*sizeof(double));
    diag2 = malloc((ny+1)*sizeof(double));
    diag3 = malloc((nx+1)*sizeof(double));
    diag4 = malloc((ny+1)*sizeof(double));

    for (i = 0; i < (nx+1); i++) {
	diag1[i] = 0.0;
	diag3[i] = 0.0;
    }
    for (i = 0; i < (ny+1); i++) {
	diag2[i] = 0.0;
	diag4[i] = 0.0;
    }
    diag1[0] = -2.0;
    diag1[1] = 1.0;
    diag2[0] = -2.0;
    diag2[1] = 1.0;

    Dxxc = CSR_sym_toeplitz(diag1, nx+1);
    Dyyc = CSR_sym_toeplitz(diag2, ny+1);

    /* set sx, sy and change Dxxc, Dyyc according to bc */
    if ((bc == 1) || (bc == 2)) {
	sx = sx0;
	sy = sy0;
    }
    else {
	sx = sx1;
	sy = sy1;
	CSRSetValue(Dxxc, 0, 1, 2.0);
	CSRSetValue(Dxxc, nx, nx-1, 2.0);
	CSRSetValue(Dyyc, 0, 1, 2.0);
	CSRSetValue(Dyyc, ny, ny-1, 2.0);
    }

    /* generate Dxx1 and Dyy1 (krons) using Dxxc and Dyyc */
    tmp = CSR_matrix_multiply(sx, Dxxc);
    Dxx1 = CSR_kronecker_product(tmp, sy);
    CSR_free(tmp);

    tmp = CSR_matrix_multiply(sy, Dyyc);
    Dyy1 = CSR_kronecker_product(sx, tmp);
    CSR_free(tmp);

    CSR_free(Dxxc);
    CSR_free(Dyyc);

    /* generate D only if requested */
    if (D != NULL) {
	*D = CSR_matrix_add_sub(Dxx1, Dyy1, 0);
    }

    // Can print matrices with this code:
    //    CSRPrint(Dyyc, "Dyyc");
    // Add "disp(Dyyc);" to plate_mat_gen.m and run Matlab to find out what they should be

    Dxx = CSR_sym_toeplitz(diag1, nx+1);
    Dyy = CSR_sym_toeplitz(diag2, ny+1);

    /*
     * alter Dxx and Dyy if bc is 1
     * set sx2, sy3, sx31, sx32, sy21, sy22 according to bc
     */
    switch (bc) {
    case 1:
	CSRSetValue(Dxx, 0, 1, 2.0);
	CSRSetValue(Dxx, nx, nx-1, 2.0);
	CSRSetValue(Dyy, 0, 1, 2.0);
	CSRSetValue(Dyy, ny, ny-1, 2.0);
	sx2 = sx1;
	sy3 = sy1;
	sx31 = sx1;
	sx32 = sx1;
	sy21 = sy1;
	sy22 = sy1;
	break;
    case 2:
	sx2 = sx0;
	sy3 = sy0;
	sx31 = sx0;
	sx32 = sx0;
	sy21 = sy0;
	sy22 = sy0;
	break;
    case 3:
    case 4:
	sx2 = sx0;
	sy3 = sy0;
	sx31 = sx0;
	sx32 = sxnu;
	sy21 = synu;
	sy22 = sy0;
	break;
    }

    /* get D2 (sum of 2 krons) */
    tmp = CSR_matrix_multiply(sx2, Dxx);
    tmp2 = CSR_kronecker_product(tmp, sy21);
    CSR_free(tmp);
    tmp = CSR_matrix_multiply(sy22, Dyy);
    tmp3 = CSR_kronecker_product(sx2, tmp);
    CSR_free(tmp);
    CSR_scalar_mult(tmp3, nu);
    *D2 = CSR_matrix_add_sub(tmp2, tmp3, 0);
    CSR_free(tmp2);
    CSR_free(tmp3);

    /* get D3 (sum of 2 krons) */
    tmp = CSR_matrix_multiply(sx31, Dxx);
    tmp2 = CSR_kronecker_product(tmp, sy3);
    CSR_free(tmp);
    CSR_scalar_mult(tmp2, nu);
    tmp = CSR_matrix_multiply(sy3, Dyy);
    tmp3 = CSR_kronecker_product(sx32, tmp);
    CSR_free(tmp);
    *D3 = CSR_matrix_add_sub(tmp2, tmp3, 0);
    CSR_free(tmp2);
    CSR_free(tmp3);

    CSR_free(Dxx);
    CSR_free(Dyy);

    /* compute Dx1 and Dy1 (spdiags) */
    for (i = 0; i < (nx+1); i++) {
	diag1[i] = 0.0;
	diag3[i] = 0.0;
    }
    diag1[0] = 1.0;
    diag1[1] = -1.0;
    diag3[0] = 1.0;
    tmp = CSR_toeplitz(diag1, nx+1, diag3, nx);
    Dx1 = CSR_matrix_multiply(sx, tmp);
    CSR_free(tmp);

    for (i = 0; i < (ny+1); i++) {
	diag2[i] = 0.0;
	diag4[i] = 0.0;
    }
    diag2[0] = 1.0;
    diag2[1] = -1.0;
    diag4[0] = 1.0;
    tmp = CSR_toeplitz(diag2, ny+1, diag4, ny);
    Dy1 = CSR_matrix_multiply(sy, tmp);
    CSR_free(tmp);

    CSR_free(sx1);
    CSR_free(sx0);
    CSR_free(sxnu);

    CSR_free(sy1);
    CSR_free(sy0);
    CSR_free(synu);

    /* alter Dx1 and Dy1 for bc 3 and 4 */
    if ((bc == 3) || (bc == 4)) {
	CSRSetValue(Dx1, 0, 0, 2.0);
	CSRSetValue(Dx1, nx, nx-1, -2.0);
	CSRSetValue(Dy1, 0, 0, 2.0);
	CSRSetValue(Dy1, ny, ny-1, -2.0);
    }

    /* compute Dxy1 (kron of Dx1, Dy1) */
    Dxy1 = CSR_kronecker_product(Dx1, Dy1);

    CSR_free(Dx1);
    CSR_free(Dy1);

    /* alter Dxy1 for bc 3 and 4 */
    if ((bc == 3) || (bc == 4)) {
	CSRSetValue(Dxy1, 0, 0, 4.0);
	CSRSetValue(Dxy1, ny, ny-1, -4.0);
	CSRSetValue(Dxy1, ((nx+1)*(ny+1))-1, (nx*ny)-1, 4.0);
	CSRSetValue(Dxy1, ((nx)*(ny+1)), ((nx-1)*ny), -4.0);
    }

    /* alter Dxx1, Dyy1 and Dxy1 for bc 4 */
    if (bc == 4) {
	CSR_zero_row(Dxx1, 0);
	CSR_zero_row(Dxx1, ny);
	CSR_zero_row(Dxx1, ss-1);
	CSR_zero_row(Dxx1, ss-ny-1);

	CSR_zero_row(Dyy1, 0);
	CSR_zero_row(Dyy1, ny);
	CSR_zero_row(Dyy1, ss-1);
	CSR_zero_row(Dyy1, ss-ny-1);

	CSR_zero_row(Dxy1, 0);
	CSR_zero_row(Dxy1, ny);
	CSR_zero_row(Dxy1, ss-1);
	CSR_zero_row(Dxy1, ss-ny-1);

	if (D != NULL) {
	    CSR_zero_row(*D, 0);
	    CSR_zero_row(*D, ny);
	    CSR_zero_row(*D, ss-1);
	    CSR_zero_row(*D, ss-ny-1);
	}
    }

    /* generate Dx2 and Dy2 (spdiags) */
    diag1[0] = -1.0;
    diag1[1] = 0.0;
    diag3[0] = -1.0;
    diag3[1] = 1.0;
    Dx2 = CSR_toeplitz(diag1, nx, diag3, nx+1);

    diag2[0] = -1.0;
    diag2[1] = 0.0;
    diag4[0] = -1.0;
    diag4[1] = 1.0;
    Dy2 = CSR_toeplitz(diag2, ny, diag4, ny+1);

    free(diag1);
    free(diag2);
    free(diag3);
    free(diag4);

    /* generate Dx and Dy (for energy checking) */
    *Dx = CSR_kron_mat_eye(Dx2, ny+1);
    *Dy = CSR_kron_eye_mat(Dy2, nx+1);

    /* generate Dxy2 (kron of Dx2, Dy2) */
     *Dxy2 = CSR_kronecker_product(Dx2, Dy2);

    CSR_free(Dx2);
    CSR_free(Dy2);

    /* generate DD (complicated calculation on Dxx1, D2, Dyy1, D3, Dxy1, Dxy2) */
    *DD = computeDD(Dxx1, *D2, Dyy1, *D3, Dxy1, *Dxy2, 2.0*(1.0-nu), nx, ny);

    if (!(*DD)) {
	/* optimised method failed, do it the slow way */
	tmp = CSR_matrix_multiply(Dxx1, *D2);
	tmp2 = CSR_matrix_multiply(Dyy1, *D3);
	tmp3 = CSR_matrix_add_sub(tmp, tmp2, 0);
	CSR_free(tmp);
	CSR_free(tmp2);
	tmp = CSR_matrix_multiply(Dxy1, *Dxy2);
	CSR_scalar_mult(tmp, 2.0 * (1.0 - nu));
	*DD = CSR_matrix_add_sub(tmp3, tmp, 0);
	CSR_free(tmp);
	CSR_free(tmp3);
    }

    CSR_free(Dxy1);
    CSR_free(Dxx1);
    CSR_free(Dyy1);

    /* generate scalevecx and scalevecy */
    scalevecx = malloc((nx+1)*sizeof(double));
    scalevecy = malloc((ny+1)*sizeof(double));
    for (i = 0; i < (nx+1); i++) {
	scalevecx[i] = 1.0;
    }
    for (i = 0; i < (ny+1); i++) {
	scalevecy[i] = 1.0;
    }
    scalevecx[0] = 0.5;
    scalevecx[nx] = 0.5;
    scalevecy[0] = 0.5;
    scalevecy[ny] = 0.5;

    /* generate scalevec from scalevecx and scalevecy */
    *scalevec = malloc(ss*sizeof(double));
    for (i = 0; i < (nx+1); i++) {
	for (j = 0; j < (ny+1); j++) {
	    (*scalevec)[(i * (ny+1)) + j] = scalevecx[i] * scalevecy[j];
	}
    }
    free(scalevecx);
    free(scalevecy);

    /* generate scalematx if requested */
    if (scalematx != NULL) {
	*scalematx = malloc(nx * (ny+1) * sizeof(double));
	for (i = 0; i < (nx * (ny+1)); i++) {
	    (*scalematx)[i] = 1.0;
	}
	for (i = 0; i < nx; i++) {
	    (*scalematx)[(i * (ny+1))] = 0.5;
	    (*scalematx)[(i * (ny+1)) + ny] = 0.5;
	}
    }

    /* generate scalematy if requested */
    if (scalematy != NULL) {
	*scalematy = malloc(ny * (nx+1) * sizeof(double));
	for (i = 0; i < (ny * (nx+1)); i++) {
	    (*scalematy)[i] = 1.0;
	}
	for (i = 0; i < ny; i++) {
	    (*scalematy)[i] = 0.5;
	    (*scalematy)[i + (ny * nx)] = 0.5;
	}
    }
}
