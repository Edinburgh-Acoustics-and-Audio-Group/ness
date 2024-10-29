/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Instrument
 */
#ifndef _TESTINSTRUMENT_H_
#define _TESTINSTRUMENT_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Instrument.h"

class TestInstrument : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrument();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrument>("testInstrument", &TestInstrument::testInstrument));
        return suiteOfTests;
    }

};

#endif
