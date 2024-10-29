#ifndef _PCG_H_
#define _PCG_H_

#include "banded.h"

typedef struct pcg_info_s {
    /* system size */
    int N;

    /* function called to perform preconditioning */
    void (*preconditioner)(void *dat, real *in, real *out, int n);

    /* pointer passed to preconditioner */
    void *precondData;

    /* temporary vectors, pre-allocated */
    real *r;
    real *p;
    real *omega;
    real *scg;

    real tol;
    int max_iter;
} pcg_info_t;

typedef struct bicg_info_s {
    /* pre-allocated temporary vectors */
    real *ax;
    real *r;
    real *rtilde;
    real *v;
    real *p;
    real *s;
    real *t;
} bicg_info_t;

typedef struct decomposition_s {
    CSRmatrix *L;
    CSRmatrix *U;
    real *scratch;
    real *diag;

    int bandSize;
    real *Ldense;
    real *Udense;
} decomposition_t;

/* single precision version of the decomposition */
typedef struct decomposition_sp_s {
    CSRmatrix *L;
    CSRmatrix *U;
    float *scratch;
    float *diag;

    int bandSize;
    float *Ldense;
    float *Udense;
} decomposition_sp_t;

pcg_info_t *pcgCreate(int size, void (*preconditioner)(void*, real*, real*, int),
		      void *precondData, real tol, int max_iter);
void pcgFree(pcg_info_t *pcg);
void pcgSolve(pcg_info_t *pcg, CSRmatrix *A, real *x, real *rhs);
void pcgSolve5x5(pcg_info_t *pcg, matrix_5x5_t *A, real *x, real *rhs);

pcg_info_t *pcgCreateSSE(CSRmatrix *csr, real tol, int max_iter);
void pcgFreeSSE(pcg_info_t *pcg);

pcg_info_t *pcgCreateJacobi(CSRmatrix *csr, real tol, int max_iter);
void pcgFreeJacobi(pcg_info_t *pcg);

pcg_info_t *pcgCreateCholesky(CSRmatrix *csr, real tol, int max_iter);
void pcgFreeCholesky(pcg_info_t *pcg);

decomposition_t *choleskyDecomposition(CSRmatrix *A);
void freeDecomposition(decomposition_t *dec);
void forwardSolve(decomposition_t *dec, real *rhs, real *x);
void backwardSolve(decomposition_t *dec, real *rhs, real *x);

/* both forward and back solves combined */
void triangularSolve(decomposition_t *dec, real *b, real *x);

real *invertMatrix(CSRmatrix *A);
CSRmatrix *invertMatrixSparse(CSRmatrix *A);

pcg_info_t *pcgCreateDenseInverse(CSRmatrix *A, real tol, int max_iter);
void pcgFreeDenseInverse(pcg_info_t *pcg);

pcg_info_t *pcgCreateBlockDenseInverse(CSRmatrix *A, int blockSize, real tol, int max_iter);
void pcgFreeBlockDenseInverse(pcg_info_t *pcg);

pcg_info_t *pcgCreateSpai(CSRmatrix *spai, real tol, int max_iter);

bicg_info_t *bicgCreate(int size);
int biCGStab(bicg_info_t *bicg, CSRmatrix* A, double* x, double* b, int maxits, double tol);
void bicgFree(bicg_info_t *bicg);

#endif
