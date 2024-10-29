/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction_StrumGenMulti
 */
#ifndef _TESTMATLABFUNCTIONSTRUMGENMULTI_H_
#define _TESTMATLABFUNCTIONSTRUMGENMULTI_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction_StrumGenMulti.h"

class TestMatlabFunction_StrumGenMulti : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction_StrumGenMulti();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunction_StrumGenMultiTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction_StrumGenMulti>("testMatlabFunction_StrumGenMulti", &TestMatlabFunction_StrumGenMulti::testMatlabFunction_StrumGenMulti));
        return suiteOfTests;
    }
};

#endif
