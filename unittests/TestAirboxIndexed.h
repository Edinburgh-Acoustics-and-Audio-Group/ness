/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for AirboxIndexed
 */
#ifndef _TESTAIRBOXINDEXED_H_
#define _TESTAIRBOXINDEXED_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "AirboxIndexed.h"

class TestAirboxIndexed : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testAirboxIndexed();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("AirboxIndexedTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestAirboxIndexed>("testAirboxIndexed", &TestAirboxIndexed::testAirboxIndexed));
        return suiteOfTests;
    }

 private:
    AirboxIndexed *airbox;
};

#endif
