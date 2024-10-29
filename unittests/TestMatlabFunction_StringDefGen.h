/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction_StringDefGen
 */
#ifndef _TESTMATLABFUNCTIONSTRINGDEFGEN_H_
#define _TESTMATLABFUNCTIONSTRINGDEFGEN_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction_StringDefGen.h"

class TestMatlabFunction_StringDefGen : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction_StringDefGen();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunction_StringDefGenTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction_StringDefGen>("testMatlabFunction_StringDefGen", &TestMatlabFunction_StringDefGen::testMatlabFunction_StringDefGen));
        return suiteOfTests;
    }
};

#endif
