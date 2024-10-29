/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Unit tests for BreakpointFunction
 */
#ifndef _TESTBREAKPOINTFUNCTION_H_
#define _TESTBREAKPOINTFUNCTION_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "BreakpointFunction.h"

class TestBreakpointFunction : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testBreakpointFunction();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("BreakpointFunctionTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestBreakpointFunction>("testBreakpointFunction", &TestBreakpointFunction::testBreakpointFunction));
        return suiteOfTests;
    }
};

#endif
