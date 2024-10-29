/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Unit tests for InputModalStrike
 */
#ifndef _TESTINPUTMODALSTRIKE_H_
#define _TESTINPUTMODALSTRIKE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputModalStrike.h"

class TestInputModalStrike : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputModalStrike();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputModalStrikeTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputModalStrike>("testInputModalStrike", &TestInputModalStrike::testInputModalStrike));
        return suiteOfTests;
    }
};

#endif
