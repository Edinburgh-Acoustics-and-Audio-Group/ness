/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Output
 */
#ifndef _TESTOUTPUT_H_
#define _TESTOUTPUT_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Output.h"

class TestOutput : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testOutput();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("OutputTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestOutput>("testOutput", &TestOutput::testOutput));
        return suiteOfTests;
    }

};

#endif
