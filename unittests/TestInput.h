/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Input
 */
#ifndef _TESTINPUT_H_
#define _TESTINPUT_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Input.h"

class TestInput : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInput();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInput>("testInput", &TestInput::testInput));
        return suiteOfTests;
    }

};

#endif
