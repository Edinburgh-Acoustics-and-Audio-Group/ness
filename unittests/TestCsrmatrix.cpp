/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestCsrmatrix.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace std;

#include <unistd.h>

void TestCsrmatrix::assertMatricesEqual(CSRmatrix *m1, CSRmatrix *m2, double tolerance)
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


void TestCsrmatrix::assertFilesEqual(char *filename1, char *filename2)
{
    unsigned char *d1, *d2;
    int len1, len2;
    FILE *f1, *f2;
    int i;
    f1 = fopen(filename1, "rb");
    CPPUNIT_ASSERT(f1 != NULL);
    f2 = fopen(filename2, "rb");
    CPPUNIT_ASSERT(f2 != NULL);
    fseek(f1, 0, SEEK_END);
    len1 = ftell(f1);
    fseek(f1, 0, SEEK_SET);
    fseek(f2, 0, SEEK_END);
    len2 = ftell(f2);
    fseek(f2, 0, SEEK_SET);
    CPPUNIT_ASSERT_EQUAL(len1, len2);
    d1 = new unsigned char[len1];
    d2 = new unsigned char[len2];
    fread(d1, 1, len1, f1);
    fread(d2, 1, len2, f2);
    fclose(f1);
    fclose(f2);

    for (i = 0; i < len1; i++) {
	CPPUNIT_ASSERT_EQUAL(d1[i], d2[i]);
    }
    delete[] d1;
    delete[] d2;
}

void TestCsrmatrix::setUp()
{
}

void TestCsrmatrix::tearDown()
{
}

CSRmatrix *TestCsrmatrix::getTestMatrix(int N, int M, double density)
{
    CSRmatrix *result = NULL;

    /* allocate result matrix */
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    if (!result) {
        return NULL;
    }
    if (density <= 0.0 || density > 20.0) {
        return NULL;
    }
    int nnz = (int)(1+(double)(N*M)/(100.0/density));
    if (nnz>N*M) nnz = N*M;
    CSR_setup(result, N, M, nnz); // At least as many non-zeroes as A or B

    srand(time(NULL));

    int idx = 0;
    int rownnz;
    result->rowStart[0] = 0;
    for (int i=0; i<N; i++) {
        rownnz = 0;
        for (int j=0; j<M; j++) {
            if (100.0*(double)rand() / (double)RAND_MAX< density && nnz >0) {
                //printf("i,j %d %d\n", i,j);
                rownnz++;
                result->colIndex[idx] = j;
                result->values[idx] = ((double)rand()) / (((double)RAND_MAX)/1000.0);
                idx++;
                nnz--;
            }
        }

        result->rowStart[i+1] = result->rowStart[i]+rownnz;
    }

    return result;
}

double* TestCsrmatrix::getTestVector(int N) {
    srand(time(NULL));
    double* result = (double *)malloc(N*sizeof(double));
    if (!result) {
        return NULL;
    }
    for (int i=0; i<N; i++) {
        result[i] = ((double)rand()) / (((double)RAND_MAX)/1000.0);
    }
    return result;
}

void TestCsrmatrix::testCsrmatrix()
{
    int i, j;
    // regression test for the CSR_kron_eye_mat problem
    int NPL = 100;
    int npl1 = NPL + 1;
    CSRmatrix *Dxy, *tmpDxx, *tmpDyy, *tmpDyy2;
    Dxy = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(Dxy, npl1, npl1, 3*npl1-2);
    for (i = 0; i < NPL; ++i) {
	CSRSetValue(Dxy, i, i, -2);  // Main diag
	// off diags
	CSRSetValue(Dxy, i+1, i, 1);
	CSRSetValue(Dxy, i, i+1, 1);
    }
    CSRSetValue(Dxy, NPL, NPL, -2); // Last diag
    CSRSetValue(Dxy, 0, 1, 2);
    CSRSetValue(Dxy, NPL, NPL-1, 2);
    
    tmpDxx = CSR_kron_mat_eye(Dxy, npl1);

    tmpDyy = CSR_kron_eye_mat(Dxy, npl1);
    CSRmatrix* tmpEye = CSR_create_eye(npl1);
    tmpDyy2 = CSR_kronecker_product(tmpEye, Dxy);

    // tmpDyy and tmpDyy2 should be equal
    assertMatricesEqual(tmpDyy, tmpDyy2, 1e-15);

    CSR_free(Dxy);
    CSR_free(tmpDxx);
    CSR_free(tmpDyy);
    CSR_free(tmpDyy2);

    /*
     * test CSR_create_eye
     */
    CSCmatrix *csc;
    CSRmatrix *mat1, *mat2, *mat3, *mat4;
    mat1 = CSR_create_eye(10);
    CPPUNIT_ASSERT_EQUAL(10, mat1->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat1->ncol);
    for (i = 0; i < 10; i++) {
	CPPUNIT_ASSERT_EQUAL(i, mat1->rowStart[i]);
	CPPUNIT_ASSERT_EQUAL(i, mat1->colIndex[i]);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[i], 1e-10);
    }
    CPPUNIT_ASSERT_EQUAL(10, mat1->rowStart[10]);

    /*
     * test CSRSetValue
     */
    CSRSetValue(mat1, 1, 3, 2.0);  // brand new value
    CSRSetValue(mat1, 2, 0, 3.0);  // another one
    CSRSetValue(mat1, 4, 4, 4.0);  // replace existing value
    // test whole matrix structure now
    CPPUNIT_ASSERT_EQUAL(10, mat1->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat1->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat1->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat1->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat1->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(5, mat1->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(6, mat1->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(7, mat1->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(8, mat1->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(9, mat1->rowStart[7]);
    CPPUNIT_ASSERT_EQUAL(10, mat1->rowStart[8]);
    CPPUNIT_ASSERT_EQUAL(11, mat1->rowStart[9]);
    CPPUNIT_ASSERT_EQUAL(12, mat1->rowStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, mat1->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat1->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat1->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(0, mat1->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(2, mat1->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(3, mat1->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(4, mat1->colIndex[6]);
    CPPUNIT_ASSERT_EQUAL(5, mat1->colIndex[7]);
    CPPUNIT_ASSERT_EQUAL(6, mat1->colIndex[8]);
    CPPUNIT_ASSERT_EQUAL(7, mat1->colIndex[9]);
    CPPUNIT_ASSERT_EQUAL(8, mat1->colIndex[10]);
    CPPUNIT_ASSERT_EQUAL(9, mat1->colIndex[11]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat1->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat1->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[11], 1e-12);

    /*
     * test CSRGetValue
     */
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, CSRGetValue(mat1, 0, 0), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, CSRGetValue(mat1, 5, 6), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, CSRGetValue(mat1, 1, 3), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, CSRGetValue(mat1, 4, 4), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, CSRGetValue(mat1, 9, 9), 1e-12);

    /*
     * test CSR_duplicate
     */
    mat2 = CSR_duplicate(mat1);
    assertMatricesEqual(mat1, mat2, 1e-12);  // check duplicate is equal
    CPPUNIT_ASSERT(mat1 != mat2);            // check it's not pointing to same memory
    CPPUNIT_ASSERT(mat1->rowStart != mat2->rowStart);
    CPPUNIT_ASSERT(mat1->colIndex != mat2->colIndex);
    CPPUNIT_ASSERT(mat1->values != mat2->values);

    /*
     * test CSR_compare
     */
    CPPUNIT_ASSERT(CSR_compare(mat1, mat2, 1e-12) == 0); // they should match
    CSRSetValue(mat2, 0, 9, 0.0);
    CPPUNIT_ASSERT(CSR_compare(mat1, mat2, 1e-12) == 0); // they should still match with an explicit zero entry
    CSRSetValue(mat2, 6, 6, 6.0);
    CPPUNIT_ASSERT(CSR_compare(mat1, mat2, 1e-12) != 0); // they should no longer match now
    CSRSetValue(mat2, 6, 6, 1.0);
    CPPUNIT_ASSERT(CSR_compare(mat1, mat2, 1e-12) == 0); // they should match again now
    CSRSetValue(mat2, 0, 8, 5.0);
    CPPUNIT_ASSERT(CSR_compare(mat1, mat2, 1e-12) != 0); // they should no longer match now
    CSR_free(mat2);

    /*
     * test CSR_copy
     */
    mat2 = CSR_duplicate(mat1);    // copy original matrix
    CSRSetValue(mat2, 0, 0, 8.9);
    CSRSetValue(mat2, 1, 3, 7.0);  // change some values
    CSRSetValue(mat2, 9, 9, 9.0);
    CPPUNIT_ASSERT(CSR_compare(mat1, mat2, 1e-12) != 0); // they shouldn't match now
    CSR_copy(mat1, mat2);          // copy back the original matrix
    assertMatricesEqual(mat1, mat2, 1e-12); // now they should match
    CSR_free(mat2);

    /*
     * test CSR_transpose
     */
    mat2 = CSR_transpose(mat1);   // transpose matrix
    // test whole matrix structure now
    CPPUNIT_ASSERT_EQUAL(10, mat2->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat2->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat2->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(6, mat2->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(8, mat2->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(9, mat2->rowStart[7]);
    CPPUNIT_ASSERT_EQUAL(10, mat2->rowStart[8]);
    CPPUNIT_ASSERT_EQUAL(11, mat2->rowStart[9]);
    CPPUNIT_ASSERT_EQUAL(12, mat2->rowStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, mat2->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->colIndex[6]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->colIndex[7]);
    CPPUNIT_ASSERT_EQUAL(6, mat2->colIndex[8]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->colIndex[9]);
    CPPUNIT_ASSERT_EQUAL(8, mat2->colIndex[10]);
    CPPUNIT_ASSERT_EQUAL(9, mat2->colIndex[11]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[11], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_get_full
     */
    static const double expectedFull[100] = {
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 0, 2, 0, 0, 0, 0, 0, 0,
	3, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 4, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    };
    double **full = CSR_get_full(mat1);
    for (i = 0; i < 10; i++) {
	for (j = 0; j < 10; j++) {
	    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFull[(i*10)+j], full[i][j], 1e-12);
	}
    }
    free(full);

    /*
     * test full_to_CSR
     */
    mat2 = full_to_CSR((double *)expectedFull, 10, 10);
    assertMatricesEqual(mat1, mat2, 1e-12);
    CSR_free(mat2);

    /*
     * test CSRFullPrint
     */
    CSRFullPrint(mat1, "csrfullprint.txt");
    assertFilesEqual("csrfullprint-expected.txt", "csrfullprint.txt");
    unlink("csrfullprint.txt");

    /*
     * test CSRMatlabPrint
     */
    CSRMatlabPrint(mat1, "csrmatlabprint.txt");
    assertFilesEqual("csrmatlabprint-expected.txt", "csrmatlabprint.txt");
    unlink("csrmatlabprint.txt");

    /*
     * test CSR_PGM_output
     */
    CSR_PGM_output(mat1, "csrpgmoutput");
    assertFilesEqual("csrpgmoutput-expected.pgm", "csrpgmoutput.pgm");
    unlink("csrpgmoutput.pgm");

    /*
     * test CSR_zero_row
     */
    mat2 = CSR_duplicate(mat1);
    CSR_zero_row(mat2, 1);
    for (i = 0; i < 10; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, CSRGetValue(mat2, 1, i), 1e-12);
    }
    CSR_free(mat2);

    /*
     * test CSR_zero_column
     */
    mat2 = CSR_duplicate(mat1);
    CSR_zero_column(mat2, 0);
    for (i = 0; i < 10; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, CSRGetValue(mat2, i, 0), 1e-12);
    }
    CSR_free(mat2);

    /*
     * test CSR_scalar_mult
     */
    CSR_scalar_mult(mat1, 3.0);
    CPPUNIT_ASSERT_EQUAL(12, mat1->rowStart[10]);  // nnz shouldn't have changed
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[0], 1e-12);  // check values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mat1->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, mat1->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[11], 1e-12);

    /*
     * test CSR_scalar_div
     */
    CSR_scalar_div(mat1, 3.0);
    CPPUNIT_ASSERT_EQUAL(12, mat1->rowStart[10]);  // nnz shouldn't have changed
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat1->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat1->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[11], 1e-12);

    /*
     * test other ops of CSR_scalar_computation
     */
    CSR_scalar_computation(mat1, 5.0, OP_ADD);
    CPPUNIT_ASSERT_EQUAL(12, mat1->rowStart[10]);  // nnz shouldn't have changed
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, mat1->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mat1->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mat1->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat1->values[11], 1e-12);

    CSR_scalar_computation(mat1, 5.0, OP_SUB);
    CPPUNIT_ASSERT_EQUAL(12, mat1->rowStart[10]);  // nnz shouldn't have changed
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat1->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat1->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat1->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat1->values[11], 1e-12);
    
    /*
     * test convert_CSR_to_CSC
     */
    csc = convert_CSR_to_CSC(mat1);
    CPPUNIT_ASSERT_EQUAL(10, csc->nrow);
    CPPUNIT_ASSERT_EQUAL(10, csc->ncol);
    CPPUNIT_ASSERT_EQUAL(0, csc->colStart[0]);
    CPPUNIT_ASSERT_EQUAL(2, csc->colStart[1]);
    CPPUNIT_ASSERT_EQUAL(3, csc->colStart[2]);
    CPPUNIT_ASSERT_EQUAL(4, csc->colStart[3]);
    CPPUNIT_ASSERT_EQUAL(6, csc->colStart[4]);
    CPPUNIT_ASSERT_EQUAL(7, csc->colStart[5]);
    CPPUNIT_ASSERT_EQUAL(8, csc->colStart[6]);
    CPPUNIT_ASSERT_EQUAL(9, csc->colStart[7]);
    CPPUNIT_ASSERT_EQUAL(10, csc->colStart[8]);
    CPPUNIT_ASSERT_EQUAL(11, csc->colStart[9]);
    CPPUNIT_ASSERT_EQUAL(12, csc->colStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, csc->rowIndex[0]);
    CPPUNIT_ASSERT_EQUAL(2, csc->rowIndex[1]);
    CPPUNIT_ASSERT_EQUAL(1, csc->rowIndex[2]);
    CPPUNIT_ASSERT_EQUAL(2, csc->rowIndex[3]);
    CPPUNIT_ASSERT_EQUAL(1, csc->rowIndex[4]);
    CPPUNIT_ASSERT_EQUAL(3, csc->rowIndex[5]);
    CPPUNIT_ASSERT_EQUAL(4, csc->rowIndex[6]);
    CPPUNIT_ASSERT_EQUAL(5, csc->rowIndex[7]);
    CPPUNIT_ASSERT_EQUAL(6, csc->rowIndex[8]);
    CPPUNIT_ASSERT_EQUAL(7, csc->rowIndex[9]);
    CPPUNIT_ASSERT_EQUAL(8, csc->rowIndex[10]);
    CPPUNIT_ASSERT_EQUAL(9, csc->rowIndex[11]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, csc->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, csc->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, csc->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, csc->values[11], 1e-12);
    
    /*
     * test CSC_print
     */
    CSC_print(csc, "cscprint.txt");
    assertFilesEqual("cscprint-expected.txt", "cscprint.txt");
    unlink("cscprint.txt");
    CSC_free(csc);


    /*
     * test CSR_save_petsc
     */
    CPPUNIT_ASSERT(CSR_save_petsc("csrpetsc.mat", mat1));
    assertFilesEqual("csrpetsc-expected.mat", "csrpetsc.mat");
    unlink("csrpetsc.mat");

    /*
     * test CSR_load_petsc
     */
    mat2 = CSR_load_petsc("csrpetsc-expected.mat");
    CPPUNIT_ASSERT(mat2 != NULL);
    assertMatricesEqual(mat1, mat2, 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_diagonal_scale
     */
    static const double vec1[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    mat2 = CSR_duplicate(mat1);
    CSR_diagonal_scale(mat2, (double*)vec1);
    CPPUNIT_ASSERT_EQUAL(12, mat2->rowStart[10]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, mat2->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat2->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, mat2->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mat2->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mat2->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, mat2->values[11], 1e-12);
    CSR_free(mat2);
    
    /*
     * test CSR_column_scale
     */
    mat2 = CSR_duplicate(mat1);
    CSR_column_scale(mat2, (double*)vec1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, mat2->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat2->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, mat2->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mat2->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mat2->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, mat2->values[11], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_matrix_vector_mult
     */
    double vec2[10];
    CSR_matrix_vector_mult(mat1, (double*)vec1, vec2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, vec2[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, vec2[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, vec2[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, vec2[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0, vec2[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, vec2[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, vec2[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, vec2[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, vec2[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, vec2[9], 1e-12);

    /*
     * test CSR_matrix_vector
     */
    CSR_matrix_vector(mat1, (double*)vec1, vec2, 0, OP_SUB); // use sub and don't re-init vector
    // result should now be zero
    for (i = 0; i < 10; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, vec2[i], 1e-12);
    }

    /*
     * test CSR_get_sub_matrix
     */
    mat2 = CSR_get_sub_matrix(mat1, 2, 2, 6, 6);
    CPPUNIT_ASSERT(mat2 != NULL);
    CPPUNIT_ASSERT_EQUAL(6, mat2->nrow);
    CPPUNIT_ASSERT_EQUAL(6, mat2->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat2->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(6, mat2->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(0, mat2->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->colIndex[5]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[5], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_zero_rows
     */
    static const int indices[6] = { 0, 2, 4, 5, 6, 7 };
    mat2 = CSR_duplicate(mat1);
    CSR_zero_rows(mat2, (int *)indices, 6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[11], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_zero_cols
     */
    mat2 = CSR_duplicate(mat1);
    CSR_zero_cols(mat2, (int *)indices, 6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[7], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[8], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[9], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[10], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, mat2->values[11], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_cut_rows
     */
    mat2 = CSR_duplicate(mat1);
    CSR_cut_rows(mat2, (int *)indices, 6);
    CPPUNIT_ASSERT_EQUAL(6, mat2->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat2->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat2->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(6, mat2->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(0, mat2->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(0, mat2->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(6, mat2->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->colIndex[6]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[6], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_cut_cols
     */
    mat2 = CSR_duplicate(mat1);
    CSR_cut_cols(mat2, (int*)indices, 6);
    CPPUNIT_ASSERT_EQUAL(10, mat2->nrow);
    CPPUNIT_ASSERT_EQUAL(6, mat2->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat2->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(6, mat2->rowStart[7]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->rowStart[8]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->rowStart[9]);
    CPPUNIT_ASSERT_EQUAL(7, mat2->rowStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, mat2->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(0, mat2->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(1, mat2->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(2, mat2->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(3, mat2->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(4, mat2->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(5, mat2->colIndex[6]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat2->values[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat2->values[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mat2->values[6], 1e-12);
    CSR_free(mat2);

    /*
     * test CSR_matrix_square
     */
    mat2 = CSR_matrix_square(mat1);
    CPPUNIT_ASSERT(mat2 != NULL);
    mat3 = CSR_load_petsc("csrsquare-expected.mat");
    assertMatricesEqual(mat3, mat2, 1e-10);    // check it's equal to saved result
    CSR_free(mat3);
    mat3 = CSR_matrix_multiply(mat1, mat1);
    assertMatricesEqual(mat3, mat2, 1e-10);    // check it's equal to result from CSR_matrix_multiply
    CSR_free(mat3);
    CSR_free(mat2);

    /*
     * test CSR_matrix_multiply
     */
    mat2 = CSR_create_eye(10);     // create a second test matrix
    CSR_scalar_mult(mat2, 2.0);
    CSRSetValue(mat2, 1, 5, 5.0);
    CSRSetValue(mat2, 3, 4, 7.0);
    mat3 = CSR_matrix_multiply(mat1, mat2);
    CPPUNIT_ASSERT_EQUAL(10, mat3->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat3->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat3->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat3->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(7, mat3->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(9, mat3->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(10, mat3->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(11, mat3->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(12, mat3->rowStart[7]);
    CPPUNIT_ASSERT_EQUAL(13, mat3->rowStart[8]);
    CPPUNIT_ASSERT_EQUAL(14, mat3->rowStart[9]);
    CPPUNIT_ASSERT_EQUAL(15, mat3->rowStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, mat3->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat3->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat3->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(0, mat3->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(2, mat3->colIndex[6]);
    CPPUNIT_ASSERT_EQUAL(3, mat3->colIndex[7]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[8]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[9]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->colIndex[10]);
    CPPUNIT_ASSERT_EQUAL(6, mat3->colIndex[11]);
    CPPUNIT_ASSERT_EQUAL(7, mat3->colIndex[12]);
    CPPUNIT_ASSERT_EQUAL(8, mat3->colIndex[13]);
    CPPUNIT_ASSERT_EQUAL(9, mat3->colIndex[14]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, mat3->values[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(14.0, mat3->values[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, mat3->values[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat3->values[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, mat3->values[8], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mat3->values[9], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[10], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[11], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[12], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[13], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[14], 1e-10);

    /*
     * test CSR_matrix_multiply_reuse
     */
    mat4 = CSR_duplicate(mat3);
    CSR_scalar_mult(mat4, 0.0);  // start with result structure but no values
    CSR_matrix_multiply_reuse(mat1, mat2, mat4);
    assertMatricesEqual(mat3, mat4, 1e-10); // check result is the same
    CSR_free(mat3);
    CSR_free(mat4);

    /*
     * test CSR_matrix_add
     */
    mat3 = CSR_matrix_add(mat1, mat2);
    CPPUNIT_ASSERT_EQUAL(10, mat3->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat3->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat3->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat3->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(6, mat3->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(8, mat3->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(9, mat3->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(10, mat3->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(11, mat3->rowStart[7]);
    CPPUNIT_ASSERT_EQUAL(12, mat3->rowStart[8]);
    CPPUNIT_ASSERT_EQUAL(13, mat3->rowStart[9]);
    CPPUNIT_ASSERT_EQUAL(14, mat3->rowStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, mat3->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat3->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat3->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(0, mat3->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(2, mat3->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(3, mat3->colIndex[6]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[7]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[8]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->colIndex[9]);
    CPPUNIT_ASSERT_EQUAL(6, mat3->colIndex[10]);
    CPPUNIT_ASSERT_EQUAL(7, mat3->colIndex[11]);
    CPPUNIT_ASSERT_EQUAL(8, mat3->colIndex[12]);
    CPPUNIT_ASSERT_EQUAL(9, mat3->colIndex[13]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, mat3->values[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, mat3->values[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mat3->values[8], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[9], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[10], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[11], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[12], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[13], 1e-10);
    mat4 = CSR_matrix_add(mat2, mat1);
    assertMatricesEqual(mat3, mat4, 1e-10); // order shouldn't matter
    CSR_free(mat4);

    /*
     * test CSR_matrix_add_reuse
     */
    mat4 = CSR_duplicate(mat3);
    CSR_scalar_mult(mat4, 0.0); // start with structure but no values
    CSR_matrix_add_reuse(mat1, mat2, mat4);
    assertMatricesEqual(mat3, mat4, 1e-10); // check it matches
    CSR_free(mat4);

    /*
     * test CSR_matrix_add_sub
     */
    mat4 = CSR_matrix_add_sub(mat1, mat2, OP_ADD);
    assertMatricesEqual(mat3, mat4, 1e-10);
    CSR_free(mat3);

    mat3 = CSR_matrix_add_sub(mat1, mat2, OP_SUB);
    CPPUNIT_ASSERT_EQUAL(10, mat3->nrow);
    CPPUNIT_ASSERT_EQUAL(10, mat3->ncol);
    CPPUNIT_ASSERT_EQUAL(0, mat3->rowStart[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat3->rowStart[1]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->rowStart[2]);
    CPPUNIT_ASSERT_EQUAL(6, mat3->rowStart[3]);
    CPPUNIT_ASSERT_EQUAL(8, mat3->rowStart[4]);
    CPPUNIT_ASSERT_EQUAL(9, mat3->rowStart[5]);
    CPPUNIT_ASSERT_EQUAL(10, mat3->rowStart[6]);
    CPPUNIT_ASSERT_EQUAL(11, mat3->rowStart[7]);
    CPPUNIT_ASSERT_EQUAL(12, mat3->rowStart[8]);
    CPPUNIT_ASSERT_EQUAL(13, mat3->rowStart[9]);
    CPPUNIT_ASSERT_EQUAL(14, mat3->rowStart[10]);
    CPPUNIT_ASSERT_EQUAL(0, mat3->colIndex[0]);
    CPPUNIT_ASSERT_EQUAL(1, mat3->colIndex[1]);
    CPPUNIT_ASSERT_EQUAL(3, mat3->colIndex[2]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->colIndex[3]);
    CPPUNIT_ASSERT_EQUAL(0, mat3->colIndex[4]);
    CPPUNIT_ASSERT_EQUAL(2, mat3->colIndex[5]);
    CPPUNIT_ASSERT_EQUAL(3, mat3->colIndex[6]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[7]);
    CPPUNIT_ASSERT_EQUAL(4, mat3->colIndex[8]);
    CPPUNIT_ASSERT_EQUAL(5, mat3->colIndex[9]);
    CPPUNIT_ASSERT_EQUAL(6, mat3->colIndex[10]);
    CPPUNIT_ASSERT_EQUAL(7, mat3->colIndex[11]);
    CPPUNIT_ASSERT_EQUAL(8, mat3->colIndex[12]);
    CPPUNIT_ASSERT_EQUAL(9, mat3->colIndex[13]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-5.0, mat3->values[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mat3->values[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-7.0, mat3->values[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, mat3->values[8], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[9], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[10], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[11], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[12], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, mat3->values[13], 1e-10);

    /*
     * test CSR_matrix_sub
     */
    mat4 = CSR_matrix_sub(mat1, mat2);
    assertMatricesEqual(mat3, mat4, 1e-10);
    CSR_free(mat4);
    CSR_free(mat3);

    /*
     * test CSR_toeplitz
     */
    static const double rowvec[8] = { 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0 };
    static const double colvec[8] = { 1.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0 };
    mat3 = CSR_toeplitz((double *)colvec, 8, (double *)rowvec, 8);
    for (i = 0; i < 8; i++) {
	for (j = 0; j < 8; j++) {
	    double val = CSRGetValue(mat3, i, j);
	    if (i < j) {
		// in the row part
		CPPUNIT_ASSERT_DOUBLES_EQUAL(rowvec[j - i], val, 1e-12);
	    }
	    else {
		// in the column part
		CPPUNIT_ASSERT_DOUBLES_EQUAL(colvec[i - j], val, 1e-12);
	    }
	}
    }
    CSR_free(mat3);

    /*
     * test CSR_sym_toeplitz
     */
    mat3 = CSR_toeplitz((double *)rowvec, 8, (double *)rowvec, 8);
    mat4 = CSR_sym_toeplitz((double *)rowvec, 8);
    assertMatricesEqual(mat3, mat4, 1e-12);
    CSR_free(mat3);
    CSR_free(mat4);

    /*
     * test CSR_kronecker_product
     */
    mat3 = CSR_kronecker_product(mat1, mat2);
    CPPUNIT_ASSERT_EQUAL(100, mat3->nrow);
    CPPUNIT_ASSERT_EQUAL(100, mat3->ncol);
    mat4 = CSR_load_petsc("csrkronecker-expected.mat");
    assertMatricesEqual(mat4, mat3, 1e-10);
    CSR_free(mat3);
    CSR_free(mat4);
    CSR_free(mat2);

    /*
     * test CSR_kron_eye_mat
     */
    mat3 = CSR_create_eye(10);
    mat4 = CSR_kronecker_product(mat3, mat1); // expected result
    CSR_free(mat3);
    mat3 = CSR_kron_eye_mat(mat1, 10);
    assertMatricesEqual(mat4, mat3, 1e-10);
    CSR_free(mat3);
    CSR_free(mat4);

    /*
     * test CSR_kron_mat_eye
     */
    mat3 = CSR_create_eye(10);
    mat4 = CSR_kronecker_product(mat1, mat3); // expected result
    CSR_free(mat3);
    mat3 = CSR_kron_mat_eye(mat1, 10);
    assertMatricesEqual(mat4, mat3, 1e-10);
    CSR_free(mat3);
    CSR_free(mat4);

    /*
     * test CSR_kron_toeplitz_eye
     */
    mat2 = CSR_sym_toeplitz((double *)rowvec, 8);
    mat3 = CSR_create_eye(8);
    mat4 = CSR_kronecker_product(mat2, mat3); // expected result
    CSR_free(mat3);
    mat3 = CSR_kron_toeplitz_eye((double *)rowvec, 8, 8);
    assertMatricesEqual(mat4, mat3, 1e-10);
    CSR_free(mat3);
    CSR_free(mat4);

    CSR_free(mat1);


    /*
     * test CSR_setup
     */
    mat1 = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CPPUNIT_ASSERT(CSR_setup(mat1, 15, 15, 20));
    CPPUNIT_ASSERT_EQUAL(15, mat1->nrow);
    CPPUNIT_ASSERT_EQUAL(15, mat1->ncol);
    CPPUNIT_ASSERT_EQUAL(20, mat1->nzmax);
    CPPUNIT_ASSERT(mat1->values != NULL);
    CPPUNIT_ASSERT(mat1->colIndex != NULL);
    CPPUNIT_ASSERT(mat1->rowStart != NULL);
    CSR_free(mat1);

    /*
     * test CSC_setup
     */
    csc = (CSCmatrix *)malloc(sizeof(CSCmatrix));
    CPPUNIT_ASSERT(CSC_setup(csc, 15, 15, 20));
    CPPUNIT_ASSERT_EQUAL(15, csc->nrow);
    CPPUNIT_ASSERT_EQUAL(15, csc->ncol);
    CPPUNIT_ASSERT_EQUAL(20, csc->nzmax);
    CPPUNIT_ASSERT(csc->values != NULL);
    CPPUNIT_ASSERT(csc->rowIndex != NULL);
    CPPUNIT_ASSERT(csc->colStart != NULL);
    CSC_free(csc);

    /*
     * test CSR_fast_mat_vec_mult
     */
    int N, M;
    CSRmatrix* A;
    double* x;
    double* b1;
    double* b2;
    double density;

    // Test correctness of mat vec mult
    N = 10;
    M = 20;
    density = 20.0;
    A = getTestMatrix(N, M, density);
    x = getTestVector(M);
    b1 = getTestVector(N);
    b2 = getTestVector(N);

    // First get the right answer using old method
    CSR_matrix_vector_mult(A, x, b1);

    // Now get the test answer for new method
    CSR_fast_mat_vec_mult(A, x, b2);

    // Compare
    for (i=0; i<N; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(b1[i], b2[i], 1e-12);
    }
    CSR_free(A);
    free(x);
    free(b1);
    free(b2);
    
    /*
     * test CSR_blk_diag
     */
    CSRmatrix* B;
    CSRmatrix* C;
    N = 5;
    M = 6;
    density = 20.0;
    A = getTestMatrix(N, N, density);
    B = getTestMatrix(M, M, density);

    // Check it works
    C = CSR_blk_diag(A, B);
    for (int i=0; i<N+M; i++) {
        for (int j=0; j<N+M; j++) {
            if (i<N && j<N) { // Top left
		CPPUNIT_ASSERT_DOUBLES_EQUAL(CSRGetValue(A, i, j), CSRGetValue(C, i, j), 1e-12);
            }
            if (i>=N && j>=N) { // Bottom right
                CPPUNIT_ASSERT_DOUBLES_EQUAL(CSRGetValue(B, i-N, j-N), CSRGetValue(C, i, j), 1e-12);
            }
            if (i>=N && j<N) { // Bottom left
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, CSRGetValue(C, i, j), 1e-12);
            }
            if (i<N && j>=N) { // Top right
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, CSRGetValue(C, i, j), 1e-12);
            }

        }
    }

    CSR_free(A);
    CSR_free(B);
    CSR_free(C);

    // Check it fails for non-square matrices
    A = getTestMatrix(N, M, density);
    B = getTestMatrix(M, N, density);
    C = CSR_blk_diag(A, B); // Should fail for A
    CPPUNIT_ASSERT(C == NULL);
    if (C != NULL) CSR_free(C);
    CSR_free(A);
    A = getTestMatrix(N, N, density);
    C = CSR_blk_diag(A, B); // Should fail for B
    CPPUNIT_ASSERT(C == NULL);
    if (C != NULL) CSR_free(C);
    CSR_free(A);
    CSR_free(B);

    /*
     * Test CSR_kron_diag_mat
     */
    CSRmatrix* D;
    N = M = 5;
    A = getTestMatrix(M, M, 20);
    x = getTestVector(N);

    B = CSR_kron_diag_mat(A, x, N);

    // Check it gives the right answer
    C = CSR_create_eye(N);
    for (i=0; i<N; i++) {
        CSRSetValue(C, i, i, x[i]);
    }
    D = CSR_kronecker_product(C, A);

    CPPUNIT_ASSERT_EQUAL(0, CSR_compare(B, D, 1.0e-16));

    CSR_free(A);
    CSR_free(B);
    CSR_free(C);
    CSR_free(D);
    free(x);

    /*
     * Test CSR_matrix_mult_sqr_diag
     */
    double* y;
    N = 10;

    x = getTestVector(N);
    y = getTestVector(N);
    A = CSR_create_eye(N);
    B = CSR_create_eye(N);
    for (int i=0; i<N; i++) {
        CSRSetValue(A, i, i, x[i]);
        CSRSetValue(B, i, i, y[i]);
    }
    // Check it gives the right answer
    C = CSR_matrix_mult_sqr_diag(A, B);
    D = CSR_matrix_multiply(A, B);
    CPPUNIT_ASSERT_EQUAL(0, CSR_compare(C, D, 1.0e-16));

    // Check it fails for non square/non-matching
    CSR_free(A);
    CSR_free(C);
    A = getTestMatrix(N, 2*N, 20.0);
    C = CSR_matrix_mult_sqr_diag(A, B);
    CPPUNIT_ASSERT(C == NULL);
    if (C != NULL) CSR_free(C);

    C = CSR_matrix_mult_sqr_diag(B, A);
    CPPUNIT_ASSERT(C == NULL);
    if (C != NULL) CSR_free(C);
    CSR_free(A);


    A = getTestMatrix(2*N, 2*N, 20.0);
    C = CSR_matrix_mult_sqr_diag(A, B);
    CPPUNIT_ASSERT(C == NULL);
    if (C != NULL) CSR_free(C);

    CSR_free(A);
    CSR_free(B);

    /*
     * Test CSR_zero_rows_new
     */
    N = 20;
    M = 15;
    A = getTestMatrix(N, M, 20);

    int* rowsToKeep = (int *)malloc(N*sizeof(int));
    int* rowsToKeep2 = (int *)malloc(N/2*sizeof(int));
    int nnzZero = 0;
    int count = 0;
    for (i = 0; i<N; i++) {
        if (i%2 == 0) { // Keep even rows
            rowsToKeep[i] = 1;
            rowsToKeep2[count] = i;
            count++;
            for (int j = A->rowStart[i]; j < A->rowStart[i+1]; j++) {
                nnzZero++;
            }
        } else {
            rowsToKeep[i] = 0;
        }
    }

    // Check it gives the right answer
    B = CSR_zero_rows_new(A, rowsToKeep);
    CSR_zero_rows(A, rowsToKeep2, N/2);
    CPPUNIT_ASSERT_EQUAL(0, CSR_compare(A, B, 1.0e-16));
    int nnz = B->rowStart[N];
    CPPUNIT_ASSERT_EQUAL(nnzZero, nnz);

    CSR_free(A);
    CSR_free(B);
    free(rowsToKeep);
    free(rowsToKeep2);
}
