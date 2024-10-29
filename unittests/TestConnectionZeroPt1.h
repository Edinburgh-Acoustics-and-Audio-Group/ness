/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ConnectionZeroPt1
 */
#ifndef _TESTCONNECTIONZEROPT1_H_
#define _TESTCONNECTIONZEROPT1_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ConnectionZeroPt1.h"

class TestConnectionZeroPt1 : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testConnectionZeroPt1();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ConnectionZeroPt1Test");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestConnectionZeroPt1>("testConnectionZeroPt1", &TestConnectionZeroPt1::testConnectionZeroPt1));
        return suiteOfTests;
    }

};

#endif
