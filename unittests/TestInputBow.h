/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InputBow
 */
#ifndef _TESTINPUTBOW_H_
#define _TESTINPUTBOW_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputBow.h"

class TestInputBow : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputBow();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputBowTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputBow>("testInputBow", &TestInputBow::testInputBow));
        return suiteOfTests;
    }

};

#endif
