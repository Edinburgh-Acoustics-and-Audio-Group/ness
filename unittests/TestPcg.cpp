/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestPcg.h"

#include <cstdlib>
#include <cstdio>
using namespace std;

extern "C" {
#include "csrmatrix.h"
#include "pcg.h"
#include "banded.h"

decomposition_sp_t *choleskyDecompositionSp(CSRmatrix *A);
void freeDecompositionSp(decomposition_sp_t *dec);
void forwardSolveSSESp(decomposition_sp_t *dec, float *rhs, float *x);
void backwardSolveSSESp(decomposition_sp_t *dec, float *rhs, float *x);
};

void TestPcg::setUp()
{
}

void TestPcg::tearDown()
{
}

void TestPcg::assertMatricesEqual(CSRmatrix *m1, CSRmatrix *m2, double tolerance)
{
    int i, j;

    CPPUNIT_ASSERT_EQUAL(m1->nrow, m2->nrow);
    CPPUNIT_ASSERT_EQUAL(m1->ncol, m2->ncol);

    for (i = 0; i < m1->nrow; i++) {
	CPPUNIT_ASSERT_EQUAL(m1->rowStart[i+1], m2->rowStart[i+1]);
	for (j = m1->rowStart[i]; j < m1->rowStart[i+1]; j++) {
	    CPPUNIT_ASSERT_EQUAL(m1->colIndex[j], m2->colIndex[j]);
	    CPPUNIT_ASSERT_DOUBLES_EQUAL(m1->values[j], m2->values[j], tolerance);
	}
    }
}

#define DOUBLE_RAND (((double)rand()) / ((double)RAND_MAX))

void TestPcg::testPcg()
{
    int i, j;

    //
    // Test unpreconditioned conjugate gradient solve
    //
    // create a 100x100 matrix
    CSRmatrix *A = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(A, 100, 100, 200);

    // fill in the diagonal
    for (i = 0; i < 100; i++) {
	CSRSetValue(A, i, i, DOUBLE_RAND * 100.0);
    }

    // fill in some smaller random values elsewhere
    for (i = 0; i < 50; i++) {
	int row = rand() % 100;
	int col = rand() % 100;
	CSRSetValue(A, row, col, DOUBLE_RAND);
	// make sure it stays symmetric
	CSRSetValue(A, col, row, DOUBLE_RAND);
    }

    // generate a random vector
    double *x, *b, *result;
    x = new double[100];
    b = new double[100];
    result = new double[100];

    for (i = 0; i < 100; i++) {
	x[i] = DOUBLE_RAND;
	result[i] = 0.0;
    }

    // multiply to give result
    CSR_matrix_vector_mult(A, x, b);

    // create a CG solver
    pcg_info_t *pcg = pcgCreate(100, NULL, NULL, 1e-12, 500);

    // do solve
    pcgSolve(pcg, A, result, b);

    // check result
    for (i = 0; i < 100; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(x[i], result[i], 1e-6);
    }

    pcgFree(pcg);

    delete[] x;
    delete[] b;
    delete[] result;
    CSR_free(A);

    //
    // Test preconditioned conjugate gradient solve, using SSE and banded matrices
    //
    // load the matrix used for generating the preconditioner
    CSRmatrix *DDDD_F = CSR_load_petsc("pcg_DDDD_F.mat");
    // load the system matrix
    A = CSR_load_petsc("pcg_A.mat");
    // create vectors
    x = new double[A->nrow];
    b = new double[A->nrow];
    result = new double[A->nrow];
    // load the RHS and the expected output
    FILE *f = fopen("pcg_x.bin", "rb");
    fread(x, 1, A->nrow * sizeof(double), f);
    fclose(f);
    f = fopen("pcg_b.bin", "rb");
    fread(b, 1, A->nrow * sizeof(double), f);
    fclose(f);
    
    // create solver
    pcg = pcgCreateSSE(DDDD_F, 1e-6, 500);
    // create 5x5 version of system matrix
    matrix_5x5_t *A5 = allocate5x5(A->nrow, 11);
    CPPUNIT_ASSERT(csrTo5x5(A, A5) != 0);
    
    // solve
    for (i = 0; i < A->nrow; i++) result[i] = 0.0;
    pcgSolve5x5(pcg, A5, result, b);

    // check result
    for (i = 0; i < A->nrow; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(x[i], result[i], 1e-6);
    }
    // free everything
    pcgFreeSSE(pcg);
    free5x5(A5);
    CSR_free(A);

    //
    // Test Cholesky decomposition
    //
    decomposition_t *decomp = choleskyDecomposition(DDDD_F);
    CSRmatrix *L = decomp->L;
    CSRmatrix *U = decomp->U;
    // The decomposition should have the following properties:
    // 1. L is lower triangular
    for (i = 0; i < L->nrow; i++) {
	for (j = L->rowStart[i]; j < L->rowStart[i+1]; j++) {
	    CPPUNIT_ASSERT(L->colIndex[j] <= i);
	}
    }
    // 2. U is the transpose of L
    CSRmatrix *Ut = CSR_transpose(U);
    assertMatricesEqual(L, Ut, 1e-12);
    CSR_free(Ut);
    // 3. L*U gives the original matrix
    CSRmatrix *LU = CSR_matrix_multiply(L, U);
    CPPUNIT_ASSERT(CSR_compare(DDDD_F, LU, 1e-8) == 0);
    CSR_free(LU);

    // SSE Cholesky was tested indirectly above. Its format is opaque and only useful to
    // the SSE triangular solves and may change if they change, so don't test it directly
    // here.

    //
    // Test C triangular solve
    //
    for (i = 0; i < DDDD_F->nrow; i++) {
	x[i] = DOUBLE_RAND;
	result[i] = 0.0;
    }
    CSR_matrix_vector_mult(DDDD_F, x, b);
    triangularSolve(decomp, b, result);
    for (i = 0; i < DDDD_F->nrow; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(x[i], result[i], 1e-6);
    }

    //
    // Test SSE triangular solve
    //
    decomposition_sp_t *decsse = choleskyDecompositionSp(DDDD_F);
    for (i = 0; i < DDDD_F->nrow; i++) {
	x[i] = DOUBLE_RAND;
	result[i] = 0.0;
    }
    CSR_matrix_vector_mult(DDDD_F, x, b);

    float *sx, *srhs, *sscratch;
    sx = new float[DDDD_F->nrow];
    srhs = new float[DDDD_F->nrow];
    sscratch = new float[DDDD_F->nrow];

    // convert to single precision
    for (i = 0; i < DDDD_F->nrow; i++) {
	srhs[i] = (float)b[i];
    }
    forwardSolveSSESp(decsse, srhs, sscratch);
    backwardSolveSSESp(decsse, sscratch, sx);

    for (i = 0; i < DDDD_F->nrow; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(x[i], (double)sx[i], 1e-5);
    }
    
    freeDecompositionSp(decsse);

    delete[] sx;
    delete[] srhs;
    delete[] sscratch;

    freeDecomposition(decomp);

    //
    // Test matrix inversion
    //
    // invert DDDD_F
    double *inv = invertMatrix(DDDD_F);
    // multiply a random vector by DDDD_F
    for (i = 0; i < DDDD_F->nrow; i++) {
	x[i] = DOUBLE_RAND;
    }
    CSR_matrix_vector_mult(DDDD_F, x, b);
    // now multiply the result by the inverse
    for (i = 0; i < DDDD_F->nrow; i++) {
	result[i] = 0.0;
	for (j = 0; j < DDDD_F->nrow; j++) {
	    result[i] += inv[(i*DDDD_F->nrow)+j] * b[j];
	}
    }
    // should match original x
    for (i = 0; i < DDDD_F->nrow; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(x[i], result[i], 1e-12);
    }

    CSR_free(DDDD_F);
    free(inv);

    delete[] x;
    delete[] b;
    delete[] result;

    //
    // Test sparse matrix inversion
    //
    int N = 10;
    A = CSR_create_eye(N);
    for (i = 0; i < N; i++) {
	CSRSetValue(A, i, i, DOUBLE_RAND);
	if (i > 0) {
	    CSRSetValue(A, i-1, i, 2.0 * DOUBLE_RAND);
	}
	if (i < (N-1)) {
	    CSRSetValue(A, i, i+1, 2.0 * DOUBLE_RAND);
	}
    }
    CSRmatrix *B = invertMatrixSparse(A);
    inv = invertMatrix(A);
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    CPPUNIT_ASSERT_DOUBLES_EQUAL(inv[i*N+j], CSRGetValue(B, i, j), 1e-12);
	}
    }
    CSR_free(A);
    CSR_free(B);
    free(inv);

    //
    // Test bicg
    //
    A = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(A, 100, 100, 200);

    // fill in the diagonal
    for (i = 0; i < 100; i++) {
	CSRSetValue(A, i, i, DOUBLE_RAND * 100.0);
    }

    // fill in some smaller random values elsewhere
    for (i = 0; i < 100; i++) {
	int row = rand() % 100;
	int col = rand() % 100;
	CSRSetValue(A, row, col, DOUBLE_RAND);
    }

    // generate a random vector
    x = new double[100];
    b = new double[100];
    result = new double[100];

    for (i = 0; i < 100; i++) {
	x[i] = DOUBLE_RAND;
	result[i] = 0.0;
    }

    // multiply to give result
    CSR_matrix_vector_mult(A, x, b);

    // create a CG solver
    bicg_info_t *bicg = bicgCreate(100);

    // do solve
    CPPUNIT_ASSERT_EQUAL(0, biCGStab(bicg, A, result, b, 500, 1.0e-12));

    // check result
    for (i = 0; i < 100; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(x[i], result[i], 1e-6);
    }

    bicgFree(bicg);

    delete[] x;
    delete[] b;
    delete[] result;
    CSR_free(A);
}
