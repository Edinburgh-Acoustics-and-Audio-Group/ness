/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ConnectionZero
 */
#ifndef _TESTCONNECTIONZERO_H_
#define _TESTCONNECTIONZERO_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ConnectionZero.h"

class TestConnectionZero : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testConnectionZero();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ConnectionZeroTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestConnectionZero>("testConnectionZero", &TestConnectionZero::testConnectionZero));
        return suiteOfTests;
    }

};

#endif
