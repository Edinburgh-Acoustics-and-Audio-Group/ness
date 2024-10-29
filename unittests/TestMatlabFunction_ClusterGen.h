/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction_ClusterGen
 */
#ifndef _TESTMATLABFUNCTIONCLUSTERGEN_H_
#define _TESTMATLABFUNCTIONCLUSTERGEN_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction_ClusterGen.h"

class TestMatlabFunction_ClusterGen : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction_ClusterGen();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunction_ClusterGenTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction_ClusterGen>("testMatlabFunction_ClusterGen", &TestMatlabFunction_ClusterGen::testMatlabFunction_ClusterGen));
        return suiteOfTests;
    }
};

#endif
