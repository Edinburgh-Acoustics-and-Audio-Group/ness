/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for MathUtil
 */
#ifndef _TESTMATHUTIL_H_
#define _TESTMATHUTIL_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MathUtil.h"

class TestMathUtil : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMathUtil();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MathUtilTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMathUtil>("testMathUtil", &TestMathUtil::testMathUtil));
        return suiteOfTests;
    }

};

#endif
