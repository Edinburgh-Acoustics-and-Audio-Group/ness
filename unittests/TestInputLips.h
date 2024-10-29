/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for lips input
 */
#ifndef _TESTINPUTLIPS_H_
#define _TESTINPUTLIPS_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputLips.h"

class TestInputLips : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputLips();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputLipsTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputLips>("testInputLips", &TestInputLips::testInputLips));
        return suiteOfTests;
    }

};

#endif
