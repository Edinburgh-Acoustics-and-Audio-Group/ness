/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestMathUtil.h"

#include "MathUtil.h"

#include <cstdio>
using namespace std;

void TestMathUtil::setUp()
{
}

void TestMathUtil::tearDown()
{
}

static double *loadBinaryVector(char *filename, int len)
{
    FILE *f = fopen(filename, "rb");
    CPPUNIT_ASSERT(f != NULL);
    double *result = new double[len];
    fread(result, 1, len * sizeof(double), f);
    fclose(f);
    return result;
}


void TestMathUtil::testMathUtil()
{
    // test factorial
    CPPUNIT_ASSERT_DOUBLES_EQUAL(720.0, factorial(6.0), 1e-10);

    // test scalar Newton solver
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-24.9290411941204, newtonSolver(0.5, -0.4, 0.3, 100000.0, 1.0, 1.0, 0.950222048838355, NULL), 1e-12);

    // test vector Newton solver
    CSRmatrix *M0 = CSR_load_petsc("newton-M0.bin");
    newton_solver_t *newton = newtonSolverCreate(15, 1e-9, 500, 1e-9, M0);
    int sz = M0->nrow;
    double *alphan = loadBinaryVector("newton-alphan.bin", sz);
    double *eta2 = loadBinaryVector("newton-eta2.bin", sz);
    double *g = loadBinaryVector("newton-g.bin", sz);
    double *K = loadBinaryVector("newton-K.bin", sz);
    double *Mdiag = loadBinaryVector("newton-Mdiag.bin", sz);
    double *q = loadBinaryVector("newton-q.bin", sz);
    double *r = loadBinaryVector("newton-r.bin", sz);
    double *R = new double[sz];
    double *phi_ra = new double[sz];

    newtonSolverVector(newton, r, eta2, g, M0, Mdiag, q, K, alphan, phi_ra, R);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, R[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.558578980962652, R[135], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.01611603358079, R[sz-1], 1e-12);

    delete[] phi_ra;
    delete[] R;
    delete[] r;
    delete[] q;
    delete[] Mdiag;
    delete[] K;
    delete[] g;
    delete[] eta2;
    delete[] alphan;
    newtonSolverFree(newton);
    CSR_free(M0);

    // test interp1
    double x[] = { 0.0, 4.0, 10.0 };
    double v[] = { 0.0, 5.0,  0.0 };
    double xq[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    double *result = new double[10];

    interp1(x, v, xq, 3, 10, result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.25, result[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, result[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.75, result[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, result[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.1666666666666667, result[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.3333333333333333, result[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, result[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.6666666666666667, result[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8333333333333333, result[9], 1e-12);

    // test interp1_interleaved
    double xv[] = { 0.0, 0.0, 4.0, 5.0, 10.0, 0.0 };
    interp1_interleaved(xv, xq, 3, 10, result);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.25, result[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, result[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.75, result[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, result[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.1666666666666667, result[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.3333333333333333, result[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, result[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.6666666666666667, result[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8333333333333333, result[9], 1e-12);
    delete[] result;

    // test croutDecomposition
    double cm[25] = {
	1.0, 1.5, 0.0, 2.0, 0.0,
	0.0, 2.0, 3.0, 2.5, 0.2,
	0.1, 0.0, 3.0, 0.4, 0.6,
	0.3, 2.2, 1.8, 4.0, 0.8,
	0.0, 0.0, 2.3, 0.9, 5.0
    };
    double cL[25];
    double cU[25];

    double cL_expected[25] = {
	1.0, 0.0, 0.0, 0.0, 0.0,
	0.0, 2.0, 0.0, 0.0, 0.0,
	0.1, -0.15, 3.225, 0.0, 0.0,
	0.3, 1.75, -0.825, 1.311627906976744, 0.0,
	0.0, 0.0, 2.3, 0.623643410852713, 4.189420803782506
    };
    double cU_expected[25] = {
	1.0, 1.5, 0.0, 2.0, 0.0,
	0.0, 1.0, 1.5, 1.25, 0.1,
	0.0, 0.0, 1.0, 0.12015503875969, 0.190697674418605,
	0.0, 0.0, 0.0, 1.0, 0.596453900709220,
	0.0, 0.0, 0.0, 0.0, 1.0
    };
    int i;
    for (i = 0; i < 25; i++) {
	cL[i] = 0.0;
	cU[i] = 0.0;
    }

    CPPUNIT_ASSERT(croutDecomposition(cm, cL, cU, 5));

    for (i = 0; i < 25; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(cL_expected[i], cL[i], 1e-10);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(cU_expected[i], cU[i], 1e-10);
    }

    // test croutSolve
    double cx[5];
    double cy[5];
    double cb[5] = { 3.0, 3.0, 3.0, 3.0, 3.0 };
    double cx_expected[5] = {
	2.715062425054666, 0.116351837483248, 0.863573393524723,
	0.055204909360232, 0.192819355293786
    };

    croutSolve(cL, cU, cb, cx, cy, 5);

    for (i = 0; i < 5; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(cx_expected[i], cx[i], 1e-10);
    }

    // test eigenvalue solver
    double ev_mat[16] = {
	1.0, 0.5, 0.33333333333333333333, 0.25,
	0.5, 1.0, 0.66666666666666666667, 0.5,
	0.33333333333333333333, 0.66666666666666666667, 1.0, 0.75,
	0.25, 0.5, 0.75, 1.0
    };
    double ev_val[4];
    double ev_vec[16];
    getEigenvalues(4, ev_mat, ev_val, ev_vec);

    // sort results into order and check them
    double best_ev = 1000000000.0;
    int best_idx = -1;
    int j;

    double ev_expected_val[4] = {
	0.207775485918012, 0.407832884117875, 0.848229155477913, 2.536162474486202
    };

    double ev_expected_vec[16] = {
	0.069318526074278, 0.361796329835911, 0.769367037085765, 0.521893398986829,
	0.442222850107573, 0.742039806455369, 0.048636017022092, 0.501448316705362,
	0.810476380106626, 0.187714392599047, 0.300968104554783, 0.466164717820999,
	0.377838497343619, 0.532206396207444, 0.561361826396131, 0.508790056532360
    };

    for (j = 0; j < 4; j++) {
	best_ev = 1000000000.0;
	best_idx = -1;

	for (i = 0; i < 4; i++) {
	    if (ev_val[i] < best_ev) {
		best_ev = ev_val[i];
		best_idx = i;
	    }
	}

	CPPUNIT_ASSERT_DOUBLES_EQUAL(ev_expected_val[j], best_ev, 1e-10);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(ev_expected_vec[j*4+0], fabs(ev_vec[best_idx+0]), 1e-10);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(ev_expected_vec[j*4+1], fabs(ev_vec[best_idx+4]), 1e-10);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(ev_expected_vec[j*4+2], fabs(ev_vec[best_idx+8]), 1e-10);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(ev_expected_vec[j*4+3], fabs(ev_vec[best_idx+12]), 1e-10);

	ev_val[best_idx] = 100000000000.0;
    }


    // test dense matrix vector multiply
    double dmv1[16] = {
	0.0, 1.0, 2.0, 3.0,
	4.0, 5.0, 6.0, 7.0,
	8.0, 9.0, 10.0, 11.0,
	12.0, 13.0, 14.0, 15.0
    };
    double dmv2[4] = {
	93.0, 27.0, 10.0, -3.0
    };
    double dmv_result[4];
    double dmv_expected[4] = {
	152.0, 279.0, 406.0, 533.0
    };
    denseMatrixVectorMultiply(dmv1, dmv2, dmv_result, 4, 4);
    for (i = 0; i < 4; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(dmv_expected[i], dmv_result[i], 1e-10);
    }

    // test dense matrix vector multiply transposed
    double dmvt_expected[4] = {
	38.0, 546.0, 1054.0, 1562.0
    };
    denseMatrixVectorMultiplyTransposed(dmv1, dmv2, dmv_result, 4, 4);
    for (i = 0; i < 4; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(dmvt_expected[i], dmv_result[i], 1e-10);
    }

    // test dense Cholesky decomposition
    double dcd_mat[16] = {
	1.0, 0.5, 0.33333333333333333333, 0.25,
	0.5, 1.0, 0.66666666666666666667, 0.5,
	0.33333333333333333333, 0.66666666666666666667, 1.0, 0.75,
	0.25, 0.5, 0.75, 1.0
    };
    double dcd_L[16];
    double dcd_U[16];
    double dcd_expected[16] = {
	1.0, 0.5, 0.33333333333333333333, 0.25,
	0.0, 0.866025403784439, 0.577350269189626, 0.433012701892219,
	0.0, 0.0, 0.745355992499930, 0.559016994374947,
	0.0, 0.0, 0.0, 0.661437827766148
    };
    denseCholeskyDecomp(4, dcd_mat, dcd_L, dcd_U);
    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    CPPUNIT_ASSERT_DOUBLES_EQUAL(dcd_expected[(j*4)+i], dcd_L[(i*4)+j], 1e-10);
	    CPPUNIT_ASSERT_DOUBLES_EQUAL(dcd_expected[(i*4)+j], dcd_U[(i*4)+j], 1e-10);
	}
    }

    // test invert lower triangle
    double invl_mat[16] = {
	1.0, 0.0, 0.0, 0.0,
	0.5, 0.866025403784439, 0.0, 0.0,
	0.333333333333333, 0.577350269189626, 0.745355992499930, 0.0,
	0.25, 0.433012701892219, 0.559016994374947, 0.661437827766148
    };
    double invl_result[16];
    double invl_expected[16] = {
	1.0, 0.0, 0.0, 0.0,
	-0.577350269189626, 1.154700538379252, 0.0, 0.0,
	0.0, -0.894427190999916, 1.341640786499874, 0.0,
	0.0, 0.0, -1.133893419027682, 1.511857892036909
    };

    invertLowerTriangle(4, invl_mat, invl_result);
    for (i = 0; i < 16; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(invl_expected[i], invl_result[i], 1e-10);
    }

    // test transpose dense matrix
    double tdm_mat[16] = {
	0.0, 1.0, 2.0, 3.0,
	4.0, 5.0, 6.0, 7.0,
	8.0, 9.0, 10.0, 11.0,
	12.0, 13.0, 14.0, 15.0
    };
    double tdm_result[16];
    transposeDenseMatrix(4, tdm_mat, tdm_result);
    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    CPPUNIT_ASSERT_DOUBLES_EQUAL(tdm_mat[(j*4)+i], tdm_result[(i*4)+j], 1e-10);
	}
    }


    // test dense matrix matrix multiply
    double dmm1[16] = {
	0.0, 1.0, 2.0, 3.0,
	4.0, 5.0, 6.0, 7.0,
	8.0, 9.0, 10.0, 11.0,
	12.0, 13.0, 14.0, 15.0
    };
    double dmm2[16] = {
	30.0, 29.0, 28.0, 27.0,
	26.0, 25.0, 24.0, 23.0,
	22.0, 21.0, 20.0, 19.0,
	18.0, 17.0, 16.0, 15.0
    };
    double dmm_result[16];
    double dmm_expected[16] = {
	124.0, 118.0, 112.0, 106.0,
	508.0, 486.0, 464.0, 442.0,
	892.0, 854.0, 816.0, 778.0,
	1276.0, 1222.0, 1168.0, 1114.0
    };
    denseMatrixMatrixMultiply(4, dmm1, dmm2, dmm_result);
    for (i = 0; i < 16; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(dmm_expected[i], dmm_result[i], 1e-10);
    }
}

