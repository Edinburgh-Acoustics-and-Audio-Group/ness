/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ConnectionP2P
 */
#ifndef _TESTCONNECTIONP2P_H_
#define _TESTCONNECTIONP2P_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ConnectionP2P.h"

class TestConnectionP2P : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testConnectionP2P();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ConnectionP2PTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestConnectionP2P>("testConnectionP2P", &TestConnectionP2P::testConnectionP2P));
        return suiteOfTests;
    }

};

#endif
