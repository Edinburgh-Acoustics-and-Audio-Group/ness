#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef WIN32
#include <sys/time.h>
#else
#include <Windows.h>
#endif

#include "csrmatrix.h"
#include "pcg.h"
#include "banded.h"

/* borrowing internal function from csrmatrix.c */
int CSR_extend(CSRmatrix *in_CSR);

/*
 * SSE-optimised solves (from sse.c)
 */
void forwardSolveSSESp(decomposition_sp_t *dec, float *rhs, float *x);
void backwardSolveSSESp(decomposition_sp_t *dec, float *rhs, float *x);

static double cclock()
{
#ifndef WIN32
    const  double  micro = 1.0e-06;    /* Conversion constant */
    static long    start = 0L, startu;
    struct timeval tp; 	              /* Structure used by gettimeofday */
    double         wall_time;          /* To hold the result */
    
    
    if ( gettimeofday( &tp, NULL) == -1 )
	wall_time = -1.0e0;
    else if( !start ) {
	start  = tp.tv_sec;
	startu = tp.tv_usec;
	wall_time = 0.0e0;
    }
    else
	wall_time = (double) (tp.tv_sec - start) + micro*(tp.tv_usec - startu);
    
    return wall_time;
#else
	return ((double)GetTickCount64()) * 1000.0;
#endif
}

/*
 * Perform a Cholesky decomposition of the given matrix (which must be symmetric)
 * Returns a structure containing the lower and upper triangular matrices, the
 * diagonal elements, and dense banded versions of the matrices which are faster
 * to use.
 */
decomposition_t *choleskyDecomposition(CSRmatrix *A)
{
    int i, j, k, l;
    real s;
    CSRmatrix *L;
    real Ljj = 1.0;
    real val = 0.0;
    real Aijms;
    decomposition_t *dec = (decomposition_t *)malloc(sizeof(decomposition_t));

    L = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(L, A->nrow, A->nrow, A->nzmax * 10);

    dec->diag = (real *)malloc(A->nrow * sizeof(real));

    dec->bandSize = 0;

    /* make sure no zeroes on diagonals */
    for (i = 0; i < A->nrow; i++) {
	for (j = A->rowStart[i]; j < A->rowStart[i+1]; j++) {
	    if (A->colIndex[j] == i) {
		if (fabs(A->values[j]) < 1e-15) A->values[j] = 1.0;
		break;
	    }
	}
    }

    for (i = 0; i < A->nrow; i++) {
	L->rowStart[i+1] = L->rowStart[i];

	for (j = 0; j < (i+1); j++) {
	    int li, lj;
	    s = 0.0;

	    li = L->rowStart[i];
	    lj = L->rowStart[j];

	    for (k = 0; k < j; k++) {
		/* s += L[i,k] * L[j,k] */
		while ((L->colIndex[li] < k) && (li < L->rowStart[i+1])) li++;
		while ((L->colIndex[lj] < k) && (lj < L->rowStart[j+1])) lj++;
		if ((li >= L->rowStart[i+1]) || (lj >= L->rowStart[j+1])) break;
		if ((L->colIndex[li] == k) && (L->colIndex[lj] == k)) {
		    s += L->values[li] * L->values[lj];
		}
	    }
	    val = 0.0;

	    /* get A[i,j] - s */
	    Aijms = -s;
	    for (l = A->rowStart[i]; l < A->rowStart[i+1]; l++) {
		if (A->colIndex[l] == j) {
		    Aijms = A->values[l] - s;
		    break;
		}
	    }

	    if (i == j) {
		/* Lij = sqrt(A[i,i] - s) */
		val = sqrt(Aijms);
		if (val < 1e-20) val = 1.0;
	    }
	    else {
		/* Lij = 1/L[j,j] * (A[i,j] - s) */
		Ljj = 1.0;
		for (l = L->rowStart[j]; l < L->rowStart[j+1]; l++) {
		    if (L->colIndex[l] == j) {
			Ljj = L->values[l];
			break;
		    }
		}
		val = (1.0 / Ljj) * Aijms;
	    }

	    if ((val > 1e-20) || (val < -1e-20)) {
		/* insert new val into decomposition matrix */
		int idx = L->rowStart[i+1];
		L->rowStart[i+1]++;
		if (L->rowStart[i+1] >= L->nzmax) {
		    /* get more space if needed */
		    CSR_extend(L);
		}
		L->colIndex[idx] = j;
		L->values[idx] = val;

		if (i == j) dec->diag[i] = 1.0 / val;

		if ((i - j) > dec->bandSize) dec->bandSize = i - j;
	    }
	}

    }
    dec->L = L;
    dec->U = CSR_transpose(L);

    /* SSE version requires bandSize to be even */
    dec->bandSize = (dec->bandSize + 1) & 0xfffffffe;

    /*printf("Performed Cholesky decomp, original nnz=%d, decomp nnz=%d\n", A->rowStart[A->nrow],
      L->rowStart[L->nrow]);*/

    /*printf("Band size: %d\n", dec->bandSize);*/

    /* convert sparse triangular matrix to banded */
    /* SSE version alignment requires bandSize+1 instead of bandSize */

    dec->Ldense = (real *)malloc((((dec->bandSize+1) * A->nrow)+1) * sizeof(real));

    memset(dec->Ldense, 0, (dec->bandSize+1) * A->nrow * sizeof(real));
    for (i = 0; i < A->nrow; i++) {
	for (j = dec->L->rowStart[i]; j < dec->L->rowStart[i+1]; j++) {
	    int col = dec->bandSize - (i - dec->L->colIndex[j]);
	    int row = i;

	    if (i != dec->L->colIndex[j]) {
		dec->Ldense[((dec->bandSize+1)*row) + col] = dec->L->values[j];

		if ((col < 0) || (col >= dec->bandSize)) {
		    printf("Error! Out of range!\n");
		}
	    }
	}
    }

    dec->Udense = (real *)malloc((((dec->bandSize+1) * A->nrow)+1) * sizeof(real));

    memset(dec->Udense, 0, (dec->bandSize+1) * A->nrow * sizeof(real));
    for (i = 0; i < A->nrow; i++) {
	for (j = dec->U->rowStart[i]; j < dec->U->rowStart[i+1]; j++) {
	    int col = (dec->U->colIndex[j] - (i+1));
	    int row = i;

	    if (i != dec->U->colIndex[j]) {
		dec->Udense[((dec->bandSize+1)*row) + col] = dec->U->values[j];

		if ((col < 0) || (col >= dec->bandSize)) {
		    printf("Error! Out of range! (%d,%d)\n", i, col);
		}
	    }
	}
    }

    /* allocate scratch vector for the triangular solver */
    dec->scratch = (real *)malloc(A->nrow * sizeof(real));
    return dec;
}

/*
 * Re-orders the dense banded data in a single precision decomposition structure
 * so that the SSE triangular solves can be used. After this is done the structure
 * can no longer be used with the C versions of the solves!!
 */
void reorderForSSE(decomposition_sp_t *dec)
{
    int i, j;
    int idx;
    int start;
    int r0i, r1i, r2i, r3i;
    int N = dec->L->nrow;
    int bandSize = dec->bandSize;
    float *in, *out;

    in = dec->Ldense;
    out = (float *)malloc((bandSize+2) * N * sizeof(float));
    /*
     *
     * First do L
     *
     */
    idx = 0;
    /*
     * Do top 4 rows (triangle)
     */
    out[0] = in[bandSize*2];
    out[1] = in[bandSize*3]; out[2] = in[bandSize*3+1];
    out[3] = in[bandSize*4]; out[4] = in[bandSize*4+1]; out[5] = in[bandSize*4+2];
    idx = 8;

    start = bandSize - 4;
    i = 4;

    /*
     * Now do the pre-banded bit at the top
     */
    while (start > 0) {
	r0i = ((bandSize+1)*i) + start;
	r1i = ((bandSize+1)*(i+1)) + start-1;
	r2i = ((bandSize+1)*(i+2)) + start-2;
	r3i = ((bandSize+1)*(i+3)) + start-3;
	for (j = 0; j < ((bandSize-start) & 0xfffffffc); j++) {
	    out[idx]   = in[r0i++];
	    out[idx+1] = in[r1i++];
	    out[idx+2] = in[r2i++];
	    out[idx+3] = in[r3i++];
	    idx += 4;
	}

	/* handle RHS triangle */
	out[idx]   = in[r1i];
	
	out[idx+1] = in[r2i++];
	out[idx+2] = in[r2i];

	out[idx+3] = in[r3i++];
	out[idx+4] = in[r3i++];
	out[idx+5] = in[r3i];

	idx += 8;

	i += 4;
	start -= 4;
    }

    /*
     * Now do the main banded bit
     */
    while (i < (N - 4)) {
	r0i = (bandSize+1)*i;
	r1i = (bandSize+1)*(i+1);
	r2i = (bandSize+1)*(i+2);
	r3i = (bandSize+1)*(i+3);

	/* handle LHS triangle */
	out[idx] = in[r0i++];
	out[idx+4] = in[r0i++];
	out[idx+8] = in[r0i++];
	out[idx+12] = in[r0i++];

	out[idx+1] = 0.0;
	out[idx+5] = in[r1i++];
	out[idx+9] = in[r1i++];
	out[idx+13] = in[r1i++];

	out[idx+2] = 0.0;
	out[idx+6] = 0.0;
	out[idx+10] = in[r2i++];
	out[idx+14] = in[r2i++];

	out[idx+3] = 0.0;
	out[idx+7] = 0.0;
	out[idx+11] = 0.0;
	out[idx+15] = in[r3i++];
	idx += 16;

	/* handle main bit */
	for (j = 0; j < (bandSize-4); j++) {
	    out[idx]   = in[r0i++];
	    out[idx+1] = in[r1i++];
	    out[idx+2] = in[r2i++];
	    out[idx+3] = in[r3i++];
	    idx += 4;
	}

	/* handle RHS triangle */
	out[idx]   = in[r1i];

	out[idx+1] = in[r2i++];
	out[idx+2] = in[r2i];

	out[idx+3] = in[r3i++];
	out[idx+4] = in[r3i++];
	out[idx+5] = in[r3i];

	idx += 8;

	i += 4;
    }

    /*
     * Do the final rows
     */
    while (i < N) {
	r0i = (bandSize+1) * i;
	for (j = 0; j < bandSize; j++) {
	    out[idx++] = in[r0i++];
	}
	i++;
    }
    free(dec->Ldense);
    dec->Ldense = out;

    /*
     *
     * Now reorder Udense
     *
     */
    in = dec->Udense;
    out = (float *)malloc((bandSize+2) * N * sizeof(float));

    idx = 0;
    out[0] = in[(N-2)*(bandSize+1)];
    out[1] = in[(N-3)*(bandSize+1)+1];
    out[2] = in[(N-3)*(bandSize+1)];
    out[3] = in[(N-4)*(bandSize+1)+2];
    out[4] = in[(N-4)*(bandSize+1)+1];
    out[5] = in[(N-4)*(bandSize+1)];
    idx = 8;

    /*
     * Do pre-banded bit at the top
     */
    start = 4;
    i = (N - 8);

    while (start < bandSize) {
	r0i = (i * (bandSize+1)) + (start+2);
	r1i = ((i+1) * (bandSize+1)) + (start+1);
	r2i = ((i+2) * (bandSize+1)) + (start+0);
	r3i = ((i+3) * (bandSize+1)) + start-1;

	for (j = start-1; j >= 0; j--) {
	    out[idx]   = in[r0i--];
	    out[idx+1] = in[r1i--];
	    out[idx+2] = in[r2i--];
	    out[idx+3] = in[r3i--];
	    idx += 4;
	}

	/* now handle LHS triangle */
	out[idx]   = in[r2i];
	out[idx+1] = in[r1i--];
	out[idx+2] = in[r1i];
	out[idx+3] = in[r0i--];
	out[idx+4] = in[r0i--];
	out[idx+5] = in[r0i];
	idx += 8;

	i -= 4;
	start += 4;
    }

    /*
     * Now do main bit
     */
    while (i >= 0) {
	r0i = (bandSize+1)*i + (bandSize-1);
	r1i = (bandSize+1)*(i+1) + (bandSize-1);
	r2i = (bandSize+1)*(i+2) + (bandSize-1);
	r3i = (bandSize+1)*(i+3) + (bandSize-1);

	/* handle RHS triangle */
	out[idx] = 0.0;
	out[idx+4] = 0.0;
	out[idx+8] = 0.0;
	out[idx+12] = in[r0i--];

	out[idx+1] = 0.0;
	out[idx+5] = 0.0;
	out[idx+9] = in[r1i--];
	out[idx+13] = in[r1i--];

	out[idx+2] = 0.0;
	out[idx+6] = in[r2i--];
	out[idx+10] = in[r2i--];
	out[idx+14] = in[r2i--];

	out[idx+3] = in[r3i--];
	out[idx+7] = in[r3i--];
	out[idx+11] = in[r3i--];
	out[idx+15] = in[r3i--];
	idx += 16;

	/* handle main bit */
	for (j = 0; j < (bandSize-4); j++) {
	    out[idx]   = in[r0i--];
	    out[idx+1] = in[r1i--];
	    out[idx+2] = in[r2i--];
	    out[idx+3] = in[r3i--];
	    idx += 4;
	}

	/* handle LHS triangle */
	out[idx]   = in[r2i];
	out[idx+1] = in[r1i--];
	out[idx+2] = in[r1i];
	out[idx+3] = in[r0i--];
	out[idx+4] = in[r0i--];
	out[idx+5] = in[r0i];
	idx += 8;

	i -= 4;
    }

    /*
     * Do the final rows
     */
    i = (i & 3) - 1;
    while (i >= 0) {
	r0i = (bandSize+1)*i + (bandSize-1);
	for (j = 0; j < bandSize; j++) {
	    out[idx++] = in[r0i--];
	}
	i--;
    }

    free(dec->Udense);
    dec->Udense = out;
}

/*
 * Perform a Cholesky decomposition of the given matrix (which must be symmetric)
 * Returns a structure containing the lower and upper triangular matrices, the
 * diagonal elements, and dense banded versions of the matrices which are faster
 * to use. Single precision version. Reorders the decomposition for SSE after
 * creating it.
 *
 * This now computes the dense banded versions directly for speed, so don't try
 * to use the L and U sparse matrices in the decomposition structure! The L
 * matrix is there as various functions use its nrow value, the U matrix is not
 * even there at all. If you need those, use the unoptimised decomposition
 * function above instead.
 *
 * The original version took several minutes to do a decomposition for the bass
 * drum membrane at 44.1kHz. The new version is almost instantaneous!
 */
decomposition_sp_t *choleskyDecompositionSp(CSRmatrix *A)
{
    int i, j, k, l;
    real s;
    CSRmatrix *L;
    real Ljj = 1.0;
    real val = 0.0;
    real Aijms;
    int N, h, hh;
    decomposition_sp_t *dec = (decomposition_sp_t *)malloc(sizeof(decomposition_t));

    N = A->nrow;

    L = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(L, A->nrow, A->nrow, A->nzmax * 10);

    dec->diag = (float *)malloc(A->nrow * sizeof(real));

    dec->bandSize = 0;

    /* make sure no zeroes on diagonals */
    /* also determine bandSize */
    for (i = 0; i < A->nrow; i++) {
	for (j = A->rowStart[i]; j < A->rowStart[i+1]; j++) {
	    if (A->colIndex[j] == i) {
		if (fabs(A->values[j]) < 1e-15) A->values[j] = 1.0;
	    }
	    if ((i - A->colIndex[j]) > dec->bandSize) {
		dec->bandSize = i - A->colIndex[j];
	    }
	}
    }
    //printf("Bandsize: %d\n", dec->bandSize);
    h = dec->bandSize / 2;
    hh = h + h;

    dec->L = L;

    /* SSE version requires bandSize to be multiple of 4 */
    dec->bandSize = (dec->bandSize + 3) & 0xfffffffc;

    /*printf("Band size: %d\n", dec->bandSize);*/

    dec->Ldense = (float *)malloc(((dec->bandSize+1) * A->nrow) * sizeof(float));
    memset(dec->Ldense, 0, (dec->bandSize+1) * A->nrow * sizeof(float));

    dec->Udense = (float *)malloc(((dec->bandSize+1) * A->nrow) * sizeof(float));
    memset(dec->Udense, 0, (dec->bandSize+1) * A->nrow * sizeof(float));

    /*
     * Now generate dense decomposition directly!
     */
    {
	matrix_5x5_t *A5 = allocate5x5(N, h);
	int col, row;

	csrTo5x5(A, A5);

#define L(r, c) dec->Ldense[((r)*(dec->bandSize+1))+(dec->bandSize-((r)-(c)))]
#define U(r, c) dec->Udense[((r)*(dec->bandSize+1))+((c)-((r)+1))]

	/* loop over all rows */
	for (i = 0; i < N; i++) {
	    double diags = 0.0;
	    double Aval = 0.0;

	    /*
	     * Loop over columns up to this row
	     */
	    j = i - dec->bandSize;
	    if (j < 0) j = 0;
	    for (; j < i; j++) {
		int offs = i - j;
		Aval = 0.0;
		/* get the value from A */
		if (offs == 1) {
		    Aval = A5->values[(25*i)+11];
		}
		else if (offs == 2) {
		    Aval = A5->values[(25*i)+10];
		}
		else if (offs == (h-1)) {
		    Aval = A5->values[(25*i)+8];
		}
		else if (offs == h) {
		    Aval = A5->values[(25*i)+7];
		}
		else if (offs == (h+1)) {
		    Aval = A5->values[(25*i)+6];
		}
		else if (offs == hh) {
		    Aval = A5->values[(25*i)+2];
		}

		/* get the dot product */
		/*
		 * row i values are the one we're on. It has values from (i - bandSize) onwards
		 * row j has values from (j - bandSize) onwards
		 */
		s = 0.0;
		k = (i - dec->bandSize);
		if (k < 0) k = 0;
		for (; k < j; k++) {
		    s += L(i, k) * L(j, k);
		}

		/* get our value */
		if (L(j, j) == 0.0) {
		    printf("Warning! Division by 0 (Ljj) in Cholesky!\n");
		}
		val = (1.0 / L(j, j)) * (Aval - s);

		/* store it */
		L(i, j) = val;
		U(j, i) = val;

		/* add it to s for the diagonal */
		diags += (val * val);
	    }

	    /* now do diagonal */
	    Aval = A5->values[(25*i)+12];

	    if (diags > Aval) {
		printf("Warning! Negative sqrt in Cholesky!\n");
	    }

	    val = sqrt(Aval - diags);
	    if (val < 1e-20) val = 1.0;
	    L(i, i) = val;

	    if (val == 0.0) {
		printf("Warning! Division by 0 (val) in Cholesky!\n");
	    }
	    dec->diag[i] = 1.0 / val;
	}

	free5x5(A5);
    }


    reorderForSSE(dec);

    /* allocate scratch vector for the triangular solver */
    dec->scratch = (float *)malloc(A->nrow * sizeof(float));
    return dec;
}



void forwardSolve(decomposition_t *dec, real *rhs, real *x)
{
    int i, j;
    int N = dec->L->nrow;
    real val;

    int start, len;

    start = dec->bandSize;
    len = 0;

    for (i = 0; i < N; i++) {

	val = rhs[i];

	for (j = start; j < dec->bandSize; j += 2) {
	    val -= dec->Ldense[(i * (dec->bandSize+1)) + j] * x[(i - dec->bandSize) + j];
	    val -= dec->Ldense[(i * (dec->bandSize+1)) + j + 1] * x[(i - dec->bandSize) + j + 1];
	}
	if (j < dec->bandSize) {
	    val -= dec->Ldense[(i * (dec->bandSize+1)) + j] * x[(i - dec->bandSize) + j];
	}
	
	x[i] = val * dec->diag[i];

	if (start) {
	    start--;
	}
    }
}

void backwardSolve(decomposition_t *dec, real *rhs, real *x)
{
    int i, j, idx;
    int N = dec->L->nrow;
    real val;
    double *y = dec->scratch;

    int start = -1;
    int xidx;

    for (i = N-1; i >= 0; i--) {
	val = rhs[i];

	xidx = i + dec->bandSize;
	if (xidx >= N) xidx = N - 1;

	for (j = start; j >= 0; j--) {
	    val -= dec->Udense[(i * (dec->bandSize+1)) + j] * x[xidx--];
	}
	x[i] = val * dec->diag[i];

	if (start < (dec->bandSize-1)) start++;
    }
}


/* y is just a scratch vector */
void triangularSolve(decomposition_t *dec, real *b, real *x)
{
    int i, j;
    int N = dec->L->nrow;
    real val;
    real *y = dec->scratch;

    /* Forward solve Ly = b */
    for (i = 0; i < N; i++) {
	val = b[i];
	/* subtract the non-diagonal elements multiplied by already solved values */
	for (j = dec->L->rowStart[i]; j < (dec->L->rowStart[i+1] - 1); j++) {
	    val -= dec->L->values[j] * y[dec->L->colIndex[j]];
	}
	/* last element in a row has to be the diagonal */
	y[i] = val / dec->L->values[j];
    }

    /* Back solve Ux = y */
    for (i = (N-1); i >= 0; i--) {
	val = y[i];
	/* this time first element in row is diagonal */
	for (j = dec->U->rowStart[i]+1; j < dec->U->rowStart[i+1]; j++) {
	    val -= dec->U->values[j] * x[dec->U->colIndex[j]];
	}
	x[i] = val / dec->U->values[dec->U->rowStart[i]];
    }
}

void freeDecomposition(decomposition_t *dec)
{
    CSR_free(dec->L);
    CSR_free(dec->U);
    free(dec->scratch);
    free(dec->Ldense);
    free(dec->Udense);
    free(dec);
}

void freeDecompositionSp(decomposition_sp_t *dec)
{
    CSR_free(dec->L);
    free(dec->scratch);
    free(dec->Ldense);
    free(dec->Udense);
    free(dec);
}

typedef struct precond_sse_s {
    decomposition_sp_t *dec;
    float *x;
    float *rhs;
} precond_sse_t;

void preconditionerSSE(void *dat, real *in, real *out, int N)
{
    int i;
    precond_sse_t *pre = (precond_sse_t *)dat;

#ifdef USE_SSE
    for (i = 0; i < N; i++) {
	pre->rhs[i] = (float)in[i];
    }
    forwardSolveSSESp(pre->dec, pre->rhs, pre->dec->scratch);
    backwardSolveSSESp(pre->dec, pre->dec->scratch, pre->x);
    for (i = 0; i < N; i++) {
	out[i] = (real)pre->x[i];
    }
#else
    fprintf(stderr, "Error: SSE preconditioner called, and SSE support not compiled in!\n");
    exit(1);
#endif
}

/*
 * Computes the dense inverse of the given matrix and returns it as an NxN array of values
 * (Only works for symmetric matrices due to use of Cholesky decomposition)
 */
real *invertMatrix(CSRmatrix *A)
{
    real *result;
    decomposition_t *chol;
    real *x, *b;
    int i, j;
    int N = A->nrow;

    /* allocate space for the inverse */
    result = (real *)malloc(N * N * sizeof(real));
    if (!result) {
	printf("Out of memory computing inverse of matrix\n");
	return NULL;
    }

    /* create a Cholesky decomposition of the matrix */
    chol = choleskyDecomposition(A);
    
    /* create vectors to hold the RHS and result */
    x = (real *)malloc(N * sizeof(real));
    b = (real *)malloc(N * sizeof(real));

    /* loop over columns of result */
    for (i = 0; i < N; i++) {
	/* initialise x and b */
	for (j = 0; j < N; j++) {
	    x[j] = 0.0;
	    b[j] = 0.0;
	}
	b[i] = 1.0;

	/* solve Ax=b for this column */
	triangularSolve(chol, b, x);

	/* copy the column into the result */
	for (j = 0; j < N; j++) {
	    result[(j * N) + i] = x[j];
	}
    }

    /* free everything */
    freeDecomposition(chol);
    free(x);
    free(b);

    return result;
}


/**
 * Computes the sparse inverse of the given matrix (uses pcg.h/.c)
 *
 * Only works for symmetric matrices due to use of Cholesky decomposition.
 * Use with caution, the inverse of a sparse matrix can be quite dense so
 * arguably this should be used only with a highly diagonally dominant matrix as
 * input, otherwise it may be best to use the standard invertMatrix
 *
 * @param A Matrix to be inverted
 * @return The sparse inverted matrix
 */
CSRmatrix *invertMatrixSparse(CSRmatrix *A)
{
    CSRmatrix *result;
    decomposition_t *chol;
    real *x, *b;
    int *idx;
    real* value;
    int i, j;
    int N = A->nrow;

    int nnzrow, nnz;
    /* allocate space for the inverse */
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    if (!CSR_setup(result, N, N, N)) {// At least diagonal has values
        printf("Out of memory computing inverse of matrix\n");
        return NULL;
    }

    /* create a Cholesky decomposition of the matrix */
    chol = choleskyDecomposition(A);

    /* create vectors to hold the RHS and result */
    x = (real *)malloc(N * sizeof(real));
    b = (real *)malloc(N * sizeof(real));

    value = (real *)malloc(N * sizeof(real));
    idx = (int *)malloc(N * sizeof(int));

    nnz = 0;
    /* loop over columns of result */
    for (i = 0; i < N; i++) {
        result->rowStart[i] = nnz;
        /* initialise x and b */
        for (j = 0; j < N; j++) {
            x[j] = 0.0;
            b[j] = 0.0;
        }
        b[i] = 1.0;

        /* solve Ax=b for this column */
        triangularSolve(chol, b, x);

        // Loop over A
        nnzrow = 0;
        for (j = 0; j < N; j++) {
            if (x[j] != 0.0) {
                idx[nnzrow] = j;
                value[nnzrow] = x[j];
                nnzrow++;
            }
        }

        // Create row in result
        nnz += nnzrow;
        result->rowStart[i + 1] = nnz;

        while (nnz > result->nzmax) { // Check there is enough storage allocated
            if (!CSR_extend(result)) {
                printf("\nfast add: Not enough memory for sparse matrix.\n");
                return (NULL );
            }
        }

        // Create cols in result
        nnzrow = 0;
        for (j = result->rowStart[i]; j < result->rowStart[i + 1]; j++) {
            result->colIndex[j] = idx[nnzrow];
            result->values[j] = value[nnzrow];
            nnzrow++;
        }
    }

    /* free everything */
    freeDecomposition(chol);
    free(x);
    free(b);
    free(idx);
    free(value);

    return result;
}



typedef struct block_dense_inverse_s
{
    int num_blocks;
    int block_size;
    real **inverses;
} block_dense_inverse_t;


block_dense_inverse_t *createBlockDenseInverse(CSRmatrix *A, int blockSize)
{
    block_dense_inverse_t *result = (block_dense_inverse_t *)malloc(sizeof(block_dense_inverse_t));
    int nblock, N;
    int start = 0;
    int size;
    CSRmatrix *sm;
    int i = 0;

    result->block_size = blockSize;
    N = A->nrow;
    nblock = N / blockSize;
    if ((N % blockSize) != 0) nblock++;
    result->num_blocks = nblock;
    result->inverses = (real **)malloc(nblock * sizeof(real*));

    while (N > 0) {
	size = blockSize;
	if (N < size) size = N;

	sm = CSR_get_sub_matrix(A, start, start, size, size);
	result->inverses[i] = invertMatrix(sm);
	CSR_free(sm);

	N -= size;
	start += size;
	i++;
    }

    return result;
}

void preconditionerBlockDenseInverse(void *dat, real *in, real *out, int n)
{
    block_dense_inverse_t *bdi = (block_dense_inverse_t *)dat;
    int i, start, size, N;
    int j, k;
    real sum;
    
    N = n;
    start = 0;
    for (i = 0; i < bdi->num_blocks; i++) {
	size = bdi->block_size;
	if (size > N) size = N;

	for (j = 0; j < size; j++) {
	    sum = 0.0;
	    for (k = 0; k < size; k++) {
		sum += in[k + start] * bdi->inverses[i][(j * size) + k];
	    }
	    out[j + start] = sum;
	}

	N -= size;
	start += size;
    }
}


void preconditionerDenseInverse(void *dat, real *in, real *out, int n)
{
    real *M = (real *)dat;
    real sum;
    int i, j;

    for (i = 0; i < n; i++) {
	sum = 0.0;
	for (j = 0; j < n; j++) {
	    sum += in[j] * M[(i * n) + j];
	}
	out[i] = sum;
    }
}


void preconditionerJacobi(void *dat, real *in, real *out, int n)
{
    /* dat is a simple array of the inverted diagonal of the matrix */
    real *diag = (real *)dat;
    int i;
    for (i = 0; i < n; i++) {
	out[i] = in[i] * diag[i];
    }
}


void preconditionerCholesky(void *dat, real *in, real *out, int n)
{
    decomposition_t *dec = (decomposition_t *)dat;
    triangularSolve(dec, in, out);
}


void preconditionerSpai(void *dat, real *in, real *out, int n)
{
    CSRmatrix *spai = (CSRmatrix *)dat;
    CSR_matrix_vector_mult(spai, in, out);
}

void createBanded(real *inv, int N, int bandSize)
{
    int i, j;
    for (i = 0; i < N; i++) {
	for (j = 0; j < (i - bandSize); j++) {
	    inv[(i*N)+j] = 0.0;
	}
	for (j = (i + bandSize); j < N; j++) {
	    inv[(i*N)+j] = 0.0;
	}
    }
}


pcg_info_t *pcgCreateDenseInverse(CSRmatrix *A, real tol, int max_iter)
{
    real *inv;
    inv = invertMatrix(A);
    return pcgCreate(A->nrow, preconditionerDenseInverse, inv, tol, max_iter);
}

pcg_info_t *pcgCreateBlockDenseInverse(CSRmatrix *A, int blockSize, real tol, int max_iter)
{
    block_dense_inverse_t *bdi = createBlockDenseInverse(A, blockSize);
    return pcgCreate(A->nrow, preconditionerBlockDenseInverse, bdi, tol, max_iter);
}

pcg_info_t *pcgCreateSpai(CSRmatrix *spai, real tol, int max_iter)
{
    return pcgCreate(spai->nrow, preconditionerSpai, spai, tol, max_iter);
}

void pcgFreeDenseInverse(pcg_info_t *pcg)
{
    free(pcg->precondData);
    pcgFree(pcg);
}

void pcgFreeBlockDenseInverse(pcg_info_t *pcg)
{
    int i;
    block_dense_inverse_t *bdi = (block_dense_inverse_t *)pcg->precondData;
    for (i = 0; i < bdi->num_blocks; i++) {
	free(bdi->inverses[i]);
    }
    free(bdi);
    pcgFree(pcg);
}

pcg_info_t *pcgCreateCholesky(CSRmatrix *csr, real tol, int max_iter)
{
    decomposition_t *dec = choleskyDecomposition(csr);
    return pcgCreate(csr->nrow, preconditionerCholesky, dec, tol, max_iter);
}


pcg_info_t *pcgCreateSSE(CSRmatrix *csr, real tol, int max_iter)
{
    decomposition_sp_t *dec = choleskyDecompositionSp(csr);
    precond_sse_t *pre = (precond_sse_t *)malloc(sizeof(precond_sse_t));
    pre->dec = dec;
    pre->x = (float *)malloc(csr->nrow * sizeof(float));
    pre->rhs = (float *)malloc(csr->nrow * sizeof(float));
    return pcgCreate(csr->nrow, preconditionerSSE, pre, tol, max_iter);
}

void pcgFreeSSE(pcg_info_t *pcg)
{
    precond_sse_t *pre = (precond_sse_t *)pcg->precondData;
    free(pre->x);
    free(pre->rhs);
    freeDecompositionSp(pre->dec);
    free(pre);
    pcgFree(pcg);
}

/*
 * Creates a PCG solver structure for the given matrix, using Jacobi preconditioner
 */
pcg_info_t *pcgCreateJacobi(CSRmatrix *csr, real tol, int max_iter)
{
    int n = csr->nrow;
    int i, j;
    real *diag;

    diag = (real *)malloc(n * sizeof(real));
    if (!diag) return NULL;

    for (i = 0; i < n; i++) {
	diag[i] = 1.0;
    }
    for (i = 0; i < n; i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    if (csr->colIndex[j] == i) {
		/* found diagonal element */
		diag[i] = 1.0 / csr->values[j];
		break;
	    }
	}
    }

    return pcgCreate(n, preconditionerJacobi, diag, tol, max_iter);
}

void pcgFreeJacobi(pcg_info_t *pcg)
{
    free(pcg->precondData);
    pcgFree(pcg);
}

void pcgFreeCholesky(pcg_info_t *pcg)
{
    freeDecomposition((decomposition_t *)pcg->precondData);
    pcgFree(pcg);
}

/*
 * Creates a PCG solver structure for the given system size. Pass NULL as the
 * preconditioner function for unpreconditioned
 */
pcg_info_t *pcgCreate(int size, void (*preconditioner)(void*, real*, real*, int),
		      void *precondData, real tol, int max_iter)
{
    pcg_info_t *pcg;

    pcg = (pcg_info_t *)malloc(sizeof(pcg_info_t));
    if (!pcg) return NULL;

    pcg->N = size;
    pcg->preconditioner = preconditioner;
    pcg->precondData = precondData;

    pcg->r = (real *)malloc(size * sizeof(real));
    pcg->p = (real *)malloc(size * sizeof(real));
    pcg->omega = (real *)malloc(size * sizeof(real));

    if ((!pcg->r) || (!pcg->p) || (!pcg->omega)) {
	free(pcg);
	return NULL;
    }

    if (preconditioner != NULL) {
	pcg->scg = (real *)malloc(size * sizeof(real));
	if (!pcg->scg) {
	    free(pcg->r);
	    free(pcg->p);
	    free(pcg->omega);
	    free(pcg);
	    return NULL;
	}
    }

    pcg->tol = tol;
    pcg->max_iter = max_iter;

    return pcg;
}

void pcgFree(pcg_info_t *pcg)
{
    free(pcg->r);
    free(pcg->p);
    free(pcg->omega);
    if (pcg->preconditioner != NULL) {
	free(pcg->scg);
    }
    free(pcg);
}

/* utility functions */
static real dotProduct(real *v1, real *v2, int size)
{
    int i;
    real result = 0.0;
    for (i = 0; i < size; i++) {
	result += v1[i] * v2[i];
    }
    return result;
}

static void scaleVector(real *vec, int size, real alpha)
{
    int i;
    for (i = 0; i < size; i++) {
	vec[i] *= alpha;
    }
}

static void vecAxpy(real *x, real *y, int size, real alpha)
{
    int i;
    for (i = 0; i < size; i++) {
	y[i] = y[i] + alpha * x[i];
    }
}


static void vecAypx(real *x, real *y, int size, real alpha)
{
    int i;
    for (i = 0; i < size; i++) {
	y[i] = alpha * y[i] + x[i];
    }
}

void pcgSolve5x5(pcg_info_t *pcg, matrix_5x5_t *A, real *x, real *rhs)
{
    int k = 0;
    real r0 = 0.0, r1, beta, dot, alpha;
    real tol = pcg->tol * pcg->tol;
    double t1, t2;

    static int firstTime = 1;

    if (pcg->preconditioner == NULL) {
	/* non-preconditioned */

	/* r = b - Ax */
	memcpy(pcg->r, rhs, pcg->N * sizeof(real));
	// FIXME: re-enable this at some point
	//CSR_matrix_vector(A, x, pcg->r, 0, 2);

	/* p = r */
	memcpy(pcg->p, pcg->r, pcg->N * sizeof(real));

	/* r1 = r . r */
	r1 = dotProduct(pcg->r, pcg->r, pcg->N);
	r0 = r1;
	while ((r1 > tol) && (k <= pcg->max_iter)) {
	    //printf("CG iteration %d, residual %.10f\n", k, r1);

	    /* omega = Ap */
#ifdef USE_SSE
	    m5x5_vector_SSE(A, pcg->p, pcg->omega);
#else
	    m5x5_vector_mult(A, pcg->p, pcg->omega);
#endif

	    /* dot = p . omega */
	    dot = dotProduct(pcg->p, pcg->omega, pcg->N);

	    alpha = r1 / dot;

	    /* x = x + alpha.p */
	    vecAxpy(pcg->p, x, pcg->N, alpha);

	    /* r = r - alpha.omega */
	    vecAxpy(pcg->omega, pcg->r, pcg->N, -alpha);

	    /*memcpy(pcg->r, rhs, pcg->N * sizeof(real));
	      CSR_matrix_vector(A, x, pcg->r, 0, 2);*/

	    r0 = r1;

	    /* r1 = r . r */
	    r1 = dotProduct(pcg->r, pcg->r, pcg->N);

	    beta = r1 / r0;

	    /* p = r + beta.p */
	    vecAypx(pcg->r, pcg->p, pcg->N, beta);
	    k++;
	}
	//printf("CG took %d iterations, r1=%.20f\n", k, r1);
    }
    else {
	t1 = cclock();
	/* preconditioned */
#ifdef USE_SSE
	m5x5_vector_SSE(A, x, pcg->omega);
#else
	m5x5_vector_mult(A, x, pcg->omega);
#endif
	memcpy(pcg->r, rhs, pcg->N * sizeof(real));
	vecAxpy(pcg->omega, pcg->r, pcg->N, -1.0);

	r1 = dotProduct(pcg->r, pcg->r, pcg->N);
	if (r1 < tol) {
	    //printf("Bailing out of PCG straight away! r1=%.20f\n", r1);
	    return;
	}

	pcg->preconditioner(pcg->precondData, pcg->r, pcg->p, pcg->N);

	r1 = dotProduct(pcg->r, pcg->p, pcg->N);

	r0 = r1;
	alpha = 0.0;

	while ((r1 > tol) && (k <= pcg->max_iter)) {
	    //printf("PCG iteration %d, residual %.10f\n", k, r1);
	    k++;
#ifdef USE_SSE
	    m5x5_vector_SSE(A, pcg->p, pcg->omega);
#else
	    m5x5_vector_mult(A, pcg->p, pcg->omega);
#endif	   

	    dot = dotProduct(pcg->p, pcg->omega, pcg->N);

	    alpha = r1 / dot;
	    vecAxpy(pcg->p, x, pcg->N, alpha);
	    vecAxpy(pcg->omega, pcg->r, pcg->N, -alpha);

	    pcg->preconditioner(pcg->precondData, pcg->r, pcg->scg, pcg->N);

	    r0 = r1;
	    r1 = dotProduct(pcg->r, pcg->scg, pcg->N);
	    beta = r1 / r0;
	    vecAypx(pcg->scg, pcg->p, pcg->N, beta);
	}
	/*if (firstTime) {
	    printf("PCG took %d iterations, r1=%.20f\n", k, r1);
	    t2 = cclock();
	    printf("Time: %f\n", t2-t1);
	    firstTime = 0;
	    }*/
    }
}

void pcgSolve(pcg_info_t *pcg, CSRmatrix *A, real *x, real *rhs)
{
    int k = 0;
    real r0 = 0.0, r1, beta, dot, alpha;
    real tol = pcg->tol * pcg->tol;
    double t1, t2;

    static int firstTime = 1;

    if (pcg->preconditioner == NULL) {
	/* non-preconditioned */

	/* r = b - Ax */
	memcpy(pcg->r, rhs, pcg->N * sizeof(real));
	CSR_matrix_vector(A, x, pcg->r, 0, 2);

	/* p = r */
	memcpy(pcg->p, pcg->r, pcg->N * sizeof(real));

	/* r1 = r . r */
	r1 = dotProduct(pcg->r, pcg->r, pcg->N);
	r0 = r1;
	while ((r1 > tol) && (k <= pcg->max_iter)) {
	    //printf("CG iteration %d, residual %.10f\n", k, r1);

	    /* omega = Ap */
	    CSR_matrix_vector_mult(A, pcg->p, pcg->omega);

	    /* dot = p . omega */
	    dot = dotProduct(pcg->p, pcg->omega, pcg->N);

	    alpha = r1 / dot;

	    /* x = x + alpha.p */
	    vecAxpy(pcg->p, x, pcg->N, alpha);

	    /* r = r - alpha.omega */
	    vecAxpy(pcg->omega, pcg->r, pcg->N, -alpha);

	    /*memcpy(pcg->r, rhs, pcg->N * sizeof(real));
	      CSR_matrix_vector(A, x, pcg->r, 0, 2);*/

	    r0 = r1;

	    /* r1 = r . r */
	    r1 = dotProduct(pcg->r, pcg->r, pcg->N);

	    beta = r1 / r0;

	    /* p = r + beta.p */
	    vecAypx(pcg->r, pcg->p, pcg->N, beta);
	    k++;
	}
	//printf("CG took %d iterations, r1=%.20f\n", k, r1);
    }
    else {
	t1 = cclock();
	/* preconditioned */
	CSR_matrix_vector_mult(A, x, pcg->omega);

	memcpy(pcg->r, rhs, pcg->N * sizeof(real));
	vecAxpy(pcg->omega, pcg->r, pcg->N, -1.0);

	pcg->preconditioner(pcg->precondData, pcg->r, pcg->p, pcg->N);

	r1 = dotProduct(pcg->r, pcg->p, pcg->N);

	r0 = r1;
	alpha = 0.0;

	while ((r1 > tol) && (k <= pcg->max_iter)) {
	    //printf("PCG iteration %d, residual %.10f\n", k, r1);
	    k++;
	    CSR_matrix_vector_mult(A, pcg->p, pcg->omega);

	    dot = dotProduct(pcg->p, pcg->omega, pcg->N);

	    alpha = r1 / dot;
	    vecAxpy(pcg->p, x, pcg->N, alpha);
	    vecAxpy(pcg->omega, pcg->r, pcg->N, -alpha);

	    pcg->preconditioner(pcg->precondData, pcg->r, pcg->scg, pcg->N);

	    r0 = r1;
	    r1 = dotProduct(pcg->r, pcg->scg, pcg->N);
	    beta = r1 / r0;
	    vecAypx(pcg->scg, pcg->p, pcg->N, beta);
	}
	/*if (firstTime) {
	    printf("PCG took %d iterations, r1=%.20f\n", k, r1);
	    t2 = cclock();
	    printf("Time: %f\n", t2-t1);
	    firstTime = 0;
	    }*/
    }
}

/*
 *
 * Biconjugate gradient stabilised method
 *
 */
static double norm(double* vec, int len) {
    double sum = 0.0;
    int i;
    for (i=0; i<len; i++) {
        sum += vec[i]*vec[i];
    }
    return sqrt(sum);
}

bicg_info_t *bicgCreate(int size)
{
    bicg_info_t *bicg = malloc(sizeof(bicg_info_t));
    bicg->ax = malloc(size * sizeof(real));
    bicg->r = malloc(size * sizeof(real));
    bicg->rtilde = malloc(size * sizeof(real));
    bicg->v = malloc(size * sizeof(real));
    bicg->p = malloc(size * sizeof(real));
    bicg->s = malloc(size * sizeof(real));
    bicg->t = malloc(size * sizeof(real));
    return bicg;
}

void bicgFree(bicg_info_t *bicg)
{
    free(bicg->ax);
    free(bicg->r);
    free(bicg->rtilde);
    free(bicg->v);
    free(bicg->p);
    free(bicg->s);
    free(bicg->t);
    free(bicg);
}

/**
 * Solves Ax = b (x = A\b) for x using biconjugate gradient stabilised method.
 *
 * @param A Matrix
 * @param x Solution vector on return
 * @param b RHS vector
 * @param maxits Max number of iterations
 * @param tol Tolerance for solution accuracy
 * @return 0 for success, 1 for failure to converge
 */
int biCGStab(bicg_info_t *bicg, CSRmatrix* A, double* x, double* b, int maxits, double tol) {
    double res = 0.0;
    double rho1, rho2, alpha, beta, omega;
    int i,j;

    int len = A->ncol;

    double normb =  norm(b, len);
    if (normb == 0.0) normb = 1.0;

    // Calc initial residual
    CSR_matrix_vector_mult(A, x, bicg->ax);
    for (i=0; i<len; i++) {
        bicg->r[i] = b[i]-bicg->ax[i];
    }
    memcpy(bicg->rtilde, bicg->r, len*sizeof(double));

    res = norm(bicg->r, len)/normb;
    if (res <= tol) {
        maxits = 0;
        return 0;
    }
    for (i = 1; i < maxits; ++i) {
        rho1 = dotProduct(bicg->rtilde, bicg->r, len);
        if (rho1 == 0) {
            return 2;
        }
        if (i == 1) {
            memcpy(bicg->p, bicg->r, len*sizeof(double));
        } else {

            beta = (rho1/rho2)*(alpha/omega);
            for (j=0; j<len; j++) {
                bicg->p[j] = bicg->r[j]+beta*(bicg->p[j]-omega*bicg->v[j]);
            }
        }


        CSR_matrix_vector_mult(A, bicg->p, bicg->v);
        alpha = rho1/dotProduct(bicg->rtilde, bicg->v, len);
        for (j=0; j<len; j++) {
            bicg->s[j] = bicg->r[j]-alpha*bicg->v[j];
        }
        res = norm(bicg->s, len)/normb;
        if (res < tol) {
            for (j=0; j<len; j++) {
                x[j] += alpha*bicg->p[j];
            }
            maxits = i;
            return 0;
        }

        CSR_matrix_vector_mult(A, bicg->s, bicg->t);
        omega = dotProduct(bicg->t,bicg->s, len)/dotProduct(bicg->t,bicg->t, len);
        for (j=0; j<len; j++) {
            x[j] += alpha*bicg->p[j]+omega*bicg->s[j];
            bicg->r[j] = bicg->s[j]-omega*bicg->t[j];
        }
        rho2 = rho1;
        res = norm(bicg->r, len)/normb;
        if (res < tol) {
            maxits = i;
            return 0;
        }
        if (omega == 0) {
            return 1;
        }
    }
    return 1;
}
