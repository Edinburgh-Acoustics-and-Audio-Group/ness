/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestBanded.h"

#include <cstdlib>
using namespace std;

void TestBanded::assertMatricesEqual(CSRmatrix *m1, CSRmatrix *m2, double tolerance)
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


void TestBanded::setUp()
{
}

void TestBanded::tearDown()
{
}

void m3x3_vector_SSE(matrix_3x3_t *m3x3, double *in, double *out);
void m5x5_vector_SSE(matrix_5x5_t *m5x5, double *in, double *out);

void TestBanded::testBanded()
{
    int i, j;

    double *x, *b1, *b2;

    x = new double[50];
    b1 = new double[50];
    b2 = new double[50];

    // create a test matrix that can be converted to banded 3x3
    CSRmatrix *csr1 = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(csr1, 50, 50, 50*9);

    for (i = 0; i < 50; i++) {
	if (i >= 11) CSRSetValue(csr1, i, i-11, (double)((i*9)-3));
	if (i >= 10) CSRSetValue(csr1, i, i-10, (double)((i*9)-2));
	if (i >= 9) CSRSetValue(csr1, i, i-9, (double)((i*9)-1));
	if (i >= 1) CSRSetValue(csr1, i, i-1, (double)((i*9)-0));
	CSRSetValue(csr1, i, i, (double)((i*9)+1));
	if (i < 49) CSRSetValue(csr1, i, i+1, (double)((i*9)+2));
	if (i < 41) CSRSetValue(csr1, i, i+9, (double)((i*9)+3));
	if (i < 40) CSRSetValue(csr1, i, i+10, (double)((i*9)+4));
	if (i < 39) CSRSetValue(csr1, i, i+11, (double)((i*9)+5));
    }

    matrix_3x3_t *m3_1 = allocate3x3(50, 10);
    CPPUNIT_ASSERT(m3_1 != NULL);
    
    // check it can be converted to 3x3 banded
    CPPUNIT_ASSERT(csrTo3x3(csr1, m3_1) != 0);

    // test converting back to CSR
    CSRmatrix *csr2 = m3x3ToCSR(m3_1);
    assertMatricesEqual(csr1, csr2, 1e-20);
    CSR_free(csr2);

    // test matrix-vector multiply
    for (i = 0; i < 50; i++) {
	x[i] = 1.0;
	b1[i] = 0.0;
	b2[i] = 0.0;
    }
    CSR_matrix_vector_mult(csr1, x, b1);
    m3x3_vector_mult(m3_1, x, b2);
    for (i = 0; i < 50; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(b1[i], b2[i], 1e-15);
    }
    
    // test SSE version
    for (i = 0; i < 50; i++) {
	b2[i] = 0.0;
    }
    m3x3_vector_SSE(m3_1, x, b2);
    for (i = 0; i < 50; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(b1[i], b2[i], 1e-15);
    }

    // test transpose
    matrix_3x3_t *m3_2 = allocate3x3(50, 10);
    CPPUNIT_ASSERT(m3_2 != NULL);
    m3x3_transpose(m3_1, m3_2);
    csr2 = CSR_transpose(csr1);
    CSRmatrix *csr3 = m3x3ToCSR(m3_2);
    assertMatricesEqual(csr2, csr3, 1e-20);
    CSR_free(csr2);
    CSR_free(csr3);
    free3x3(m3_2);

    // test duplicate
    m3_2 = m3x3_duplicate(m3_1);
    csr2 = m3x3ToCSR(m3_2);
    assertMatricesEqual(csr2, csr1, 1e-20);
    CSR_free(csr2);
    free3x3(m3_2);

    // test matrix add
    csr2 = CSR_create_eye(50);
    m3_2 = allocate3x3(50, 10);
    csrTo3x3(csr2, m3_2);
    matrix_3x3_t *m3_3 = allocate3x3(50, 10);
    m3x3_matrix_add(m3_1, m3_2, m3_3);
    csr3 = CSR_matrix_add(csr1, csr2);
    CSRmatrix *csr4 = m3x3ToCSR(m3_3);
    assertMatricesEqual(csr3, csr4, 1e-15);
    CSR_free(csr2);
    CSR_free(csr3);
    CSR_free(csr4);
    free3x3(m3_2);
    free3x3(m3_3);

    // test scalar multiply
    m3x3_scalar_mult(m3_1, 5.0);
    CSR_scalar_mult(csr1, 5.0);
    csr2 = m3x3ToCSR(m3_1);
    assertMatricesEqual(csr1, csr2, 1e-20);
    CSR_free(csr2);

    // add extra value to csr1 and check it CAN'T be converted
    CSRSetValue(csr1, 0, 48, 1.0);
    CPPUNIT_ASSERT(csrTo3x3(csr1, m3_1) == 0);

    CSR_free(csr1);
    free3x3(m3_1);


    // now do 5x5 tests
    // create a 5x5 banded matrix and convert
    csr1 = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(csr1, 50, 50, 50*9);

    for (i = 0; i < 50; i++) {

	if (i >= 22) CSRSetValue(csr1, i, i-22, (double)((i*25)-11));
	if (i >= 21) CSRSetValue(csr1, i, i-21, (double)((i*25)-10));
	if (i >= 20) CSRSetValue(csr1, i, i-20, (double)((i*25)-9));
	if (i >= 19) CSRSetValue(csr1, i, i-19, (double)((i*25)-8));
	if (i >= 18) CSRSetValue(csr1, i, i-18, (double)((i*25)-7));

	if (i >= 12) CSRSetValue(csr1, i, i-12, (double)((i*25)-6));
	if (i >= 11) CSRSetValue(csr1, i, i-11, (double)((i*25)-5));
	if (i >= 10) CSRSetValue(csr1, i, i-10, (double)((i*25)-4));
	if (i >= 9) CSRSetValue(csr1, i, i-9, (double)((i*25)-3));
	if (i >= 8) CSRSetValue(csr1, i, i-8, (double)((i*25)-2));

	if (i >= 2) CSRSetValue(csr1, i, i-2, (double)((i*25)-1));
	if (i >= 1) CSRSetValue(csr1, i, i-1, (double)((i*25)-0));
	CSRSetValue(csr1, i, i, (double)((i*25)+1));
	if (i < 49) CSRSetValue(csr1, i, i+1, (double)((i*25)+2));
	if (i < 48) CSRSetValue(csr1, i, i+1, (double)((i*25)+3));

	if (i < 42) CSRSetValue(csr1, i, i+8, (double)((i*25)+4));
	if (i < 41) CSRSetValue(csr1, i, i+9, (double)((i*25)+5));
	if (i < 40) CSRSetValue(csr1, i, i+10, (double)((i*25)+6));
	if (i < 39) CSRSetValue(csr1, i, i+11, (double)((i*25)+7));
	if (i < 38) CSRSetValue(csr1, i, i+12, (double)((i*25)+8));

	if (i < 32) CSRSetValue(csr1, i, i+18, (double)((i*25)+9));
	if (i < 31) CSRSetValue(csr1, i, i+19, (double)((i*25)+10));
	if (i < 30) CSRSetValue(csr1, i, i+20, (double)((i*25)+11));
	if (i < 29) CSRSetValue(csr1, i, i+21, (double)((i*25)+12));
	if (i < 28) CSRSetValue(csr1, i, i+22, (double)((i*25)+13));
    }

    matrix_5x5_t *m5_1 = allocate5x5(50, 10);
    CPPUNIT_ASSERT(m5_1 != NULL);
    
    // check it can be converted to 5x5 banded
    CPPUNIT_ASSERT(csrTo5x5(csr1, m5_1) != 0);

    // test converting back to CSR
    csr2 = m5x5ToCSR(m5_1);
    assertMatricesEqual(csr1, csr2, 1e-20);
    CSR_free(csr2);

    // test matrix-vector multiply
    for (i = 0; i < 50; i++) {
	x[i] = 1.0;
	b1[i] = 0.0;
	b2[i] = 0.0;
    }
    CSR_matrix_vector_mult(csr1, x, b1);
    m5x5_vector_mult(m5_1, x, b2);
    for (i = 0; i < 50; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(b1[i], b2[i], 1e-15);
    }
    
    // test SSE version
    for (i = 0; i < 50; i++) {
	b2[i] = 0.0;
    }
    m5x5_vector_SSE(m5_1, x, b2);
    for (i = 0; i < 50; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(b1[i], b2[i], 1e-15);
    }

    // test duplicate
    matrix_5x5_t *m5_2 = m5x5_duplicate(m5_1);
    csr2 = m5x5ToCSR(m5_2);
    assertMatricesEqual(csr2, csr1, 1e-20);
    CSR_free(csr2);
    free5x5(m5_2);

    // test matrix add
    csr2 = CSR_create_eye(50);
    m5_2 = allocate5x5(50, 10);
    csrTo5x5(csr2, m5_2);
    matrix_5x5_t *m5_3 = allocate5x5(50, 10);
    m5x5_matrix_add(m5_1, m5_2, m5_3);
    csr3 = CSR_matrix_add(csr1, csr2);
    csr4 = m5x5ToCSR(m5_3);
    assertMatricesEqual(csr3, csr4, 1e-15);
    CSR_free(csr2);
    CSR_free(csr3);
    CSR_free(csr4);
    free5x5(m5_2);
    free5x5(m5_3);

    // test scalar multiply
    m5x5_scalar_mult(m5_1, 5.0);
    CSR_scalar_mult(csr1, 5.0);
    csr2 = m5x5ToCSR(m5_1);
    assertMatricesEqual(csr1, csr2, 1e-20);
    CSR_free(csr2);

    // add extra value to csr and check it CAN'T be converted
    CSRSetValue(csr1, 0, 48, 1.0);
    CPPUNIT_ASSERT(csrTo5x5(csr1, m5_1) == 0);

    CSR_free(csr1);
    free5x5(m5_1);


    // test multiplying 3x3 matrices to give a 5x5
    // create 3x3 banded matrix
    csr1 = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(csr1, 50, 50, 50*9);

    for (i = 0; i < 50; i++) {
	if (i >= 11) CSRSetValue(csr1, i, i-11, (double)((i*9)-3));
	if (i >= 10) CSRSetValue(csr1, i, i-10, (double)((i*9)-2));
	if (i >= 9) CSRSetValue(csr1, i, i-9, (double)((i*9)-1));
	if (i >= 1) CSRSetValue(csr1, i, i-1, (double)((i*9)-0));
	CSRSetValue(csr1, i, i, (double)((i*9)+1));
	if (i < 49) CSRSetValue(csr1, i, i+1, (double)((i*9)+2));
	if (i < 41) CSRSetValue(csr1, i, i+9, (double)((i*9)+3));
	if (i < 40) CSRSetValue(csr1, i, i+10, (double)((i*9)+4));
	if (i < 39) CSRSetValue(csr1, i, i+11, (double)((i*9)+5));
    }
    m3_1 = allocate3x3(50, 10);
    csrTo3x3(csr1, m3_1);

    // duplicate it for the other input
    m3_2 = m3x3_duplicate(m3_1);

    // allocate 5x5 matrix for the result
    m5_1 = allocate5x5(50, 10);

    m3x3_matrix_multiply(m3_1, m3_2, m5_1);

    // check against CSR library
    csr2 = CSR_matrix_multiply(csr1, csr1);
    csr3 = m5x5ToCSR(m5_1);

    assertMatricesEqual(csr2, csr3, 1e-10);
    CSR_free(csr1);
    CSR_free(csr2);
    CSR_free(csr3);
    
    free3x3(m3_1);
    free3x3(m3_2);
    free5x5(m5_1);

    delete[] x;
    delete[] b1;
    delete[] b2;
}
