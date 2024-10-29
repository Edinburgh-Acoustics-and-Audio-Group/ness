/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Airbox
 */
#ifndef _TESTAIRBOX_H_
#define _TESTAIRBOX_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Airbox.h"

class TestAirbox : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testAirbox();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("AirboxTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestAirbox>("testAirbox", &TestAirbox::testAirbox));
        return suiteOfTests;
    }

 private:
    Airbox *airbox;
};

#endif
