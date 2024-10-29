/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for PCG solver related code
 */
#ifndef _TESTPCG_H_
#define _TESTPCG_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

extern "C" {
#include "pcg.h"
#include "csrmatrix.h"
};

class TestPcg : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testPcg();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("PcgTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestPcg>("testPcg",
							       &TestPcg::testPcg));
	return suiteOfTests;
    }

 protected:
    void assertMatricesEqual(CSRmatrix *m1, CSRmatrix *m2, double tolerance);

};

#endif
