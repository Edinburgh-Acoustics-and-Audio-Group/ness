/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InputStrike
 */
#ifndef _TESTINPUTSTRIKE_H_
#define _TESTINPUTSTRIKE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputStrike.h"
#include "DummyComponent1D.h"

class TestInputStrike : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputStrike();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputStrikeTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputStrike>("testInputStrike", &TestInputStrike::testInputStrike));
        return suiteOfTests;
    }

 private:
    DummyComponent1D *component;
    InputStrike *strike;

};

#endif
