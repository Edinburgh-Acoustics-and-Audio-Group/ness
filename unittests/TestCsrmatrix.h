/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Unit tests for csrmatrix library
 */
#ifndef _TESTCSRMATRIX_H_
#define _TESTCSRMATRIX_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

extern "C" {
#include "csrmatrix.h"
};

class TestCsrmatrix : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testCsrmatrix();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CsrmatrixTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestCsrmatrix>("testCsrmatrix",
								     &TestCsrmatrix::testCsrmatrix));
	return suiteOfTests;
    }

 protected:
    void assertMatricesEqual(CSRmatrix *m1, CSRmatrix *m2, double tolerance);
    void assertFilesEqual(char *filename1, char *filename2);
    CSRmatrix *getTestMatrix(int N, int M, double density);
    double* getTestVector(int N);

};

#endif
