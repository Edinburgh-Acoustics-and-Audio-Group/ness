/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for OutputPressure
 */
#ifndef _TESTOUTPUTPRESSURE_H_
#define _TESTOUTPUTPRESSURE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "OutputPressure.h"

class TestOutputPressure : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testOutputPressure();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("OutputPressureTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestOutputPressure>("testOutputPressure", &TestOutputPressure::testOutputPressure));
        return suiteOfTests;
    }

};

#endif
