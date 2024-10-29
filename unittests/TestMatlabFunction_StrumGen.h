/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction_StrumGen
 */
#ifndef _TESTMATLABFUNCTIONSTRUMGEN_H_
#define _TESTMATLABFUNCTIONSTRUMGEN_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction_StrumGen.h"

class TestMatlabFunction_StrumGen : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction_StrumGen();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunction_StrumGenTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction_StrumGen>("testMatlabFunction_StrumGen", &TestMatlabFunction_StrumGen::testMatlabFunction_StrumGen));
        return suiteOfTests;
    }
};

#endif
