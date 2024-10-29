/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InputSample
 */
#ifndef _TESTINPUTSAMPLE_H_
#define _TESTINPUTSAMPLE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputSample.h"

class TestInputSample : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputSample();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputSampleTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputSample>("testInputSample", &TestInputSample::testInputSample));
        return suiteOfTests;
    }

};

#endif
