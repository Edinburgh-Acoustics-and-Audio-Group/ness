/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Unit tests for ConnectionNet1
 */
#ifndef _TESTCONNECTIONNET1_H_
#define _TESTCONNECTIONNET1_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ConnectionNet1.h"

class TestConnectionNet1 : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testConnectionNet1();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ConnectionNet1Test");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestConnectionNet1>("testConnectionNet1", &TestConnectionNet1::testConnectionNet1));
        return suiteOfTests;
    }

};

#endif
