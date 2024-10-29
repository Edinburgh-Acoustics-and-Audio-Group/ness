/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Unit tests for InputModalSine
 */
#ifndef _TESTINPUTMODALSINE_H_
#define _TESTINPUTMODALSINE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputModalSine.h"

class TestInputModalSine : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputModalSine();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputModalSineTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputModalSine>("testInputModalSine", &TestInputModalSine::testInputModalSine));
        return suiteOfTests;
    }
};

#endif
