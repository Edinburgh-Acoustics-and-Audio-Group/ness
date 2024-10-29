/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for OutputDifference
 */
#ifndef _TESTOUTPUTDIFFERENCE_H_
#define _TESTOUTPUTDIFFERENCE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "OutputDifference.h"

class TestOutputDifference : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testOutputDifference();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("OutputDifferenceTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestOutputDifference>("testOutputDifference", &TestOutputDifference::testOutputDifference));
        return suiteOfTests;
    }

};

#endif
