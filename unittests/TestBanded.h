/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for banded matrix library
 */
#ifndef _TESTBANDED_H_
#define _TESTBANDED_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

extern "C" {
#include "banded.h"
};

class TestBanded : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testBanded();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("BandedTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestBanded>("testBanded",
								     &TestBanded::testBanded));
	return suiteOfTests;
    }

 protected:
    void assertMatricesEqual(CSRmatrix *m1, CSRmatrix *m2, double tolerance);

};

#endif
