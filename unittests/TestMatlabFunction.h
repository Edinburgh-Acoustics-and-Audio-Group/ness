/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction
 */
#ifndef _TESTMATLABFUNCTION_H_
#define _TESTMATLABFUNCTION_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction.h"

class TestMatlabFunction : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunctionTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction>("testMatlabFunction", &TestMatlabFunction::testMatlabFunction));
        return suiteOfTests;
    }
};

#endif
