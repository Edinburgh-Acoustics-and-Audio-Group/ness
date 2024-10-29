/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InputPluck
 */
#ifndef _TESTINPUTPLUCK_H_
#define _TESTINPUTPLUCK_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputPluck.h"
#include "DummyComponent1D.h"

class TestInputPluck : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputPluck();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputPluckTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputPluck>("testInputPluck", &TestInputPluck::testInputPluck));
        return suiteOfTests;
    }

 private:
    DummyComponent1D *component;
    InputPluck *pluck;
};

#endif
