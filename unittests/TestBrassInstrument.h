/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for brass instrument
 */
#ifndef _TESTBRASSINSTRUMENT_H_
#define _TESTBRASSINSTRUMENT_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "BrassInstrument.h"

class TestBrassInstrument : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testBrassInstrument();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("BrassInstrumentTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestBrassInstrument>("testBrassInstrument", &TestBrassInstrument::testBrassInstrument));
        return suiteOfTests;
    }

};

#endif
