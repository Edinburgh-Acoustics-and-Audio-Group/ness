/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction_FretDefGen
 */
#ifndef _TESTMATLABFUNCTIONFRETDEFGEN_H_
#define _TESTMATLABFUNCTIONFRETDEFGEN_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction_FretDefGen.h"

class TestMatlabFunction_FretDefGen : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction_FretDefGen();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunction_FretDefGenTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction_FretDefGen>("testMatlabFunction_FretDefGen", &TestMatlabFunction_FretDefGen::testMatlabFunction_FretDefGen));
        return suiteOfTests;
    }
};

#endif
