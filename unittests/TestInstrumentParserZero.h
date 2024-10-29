/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InstrumentParserZero
 */
#ifndef _TESTINSTRUMENTPARSERZERO_H_
#define _TESTINSTRUMENTPARSERZERO_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserZero.h"

class TestInstrumentParserZero : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserZero();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserZeroTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserZero>("testInstrumentParserZero", &TestInstrumentParserZero::testInstrumentParserZero));
        return suiteOfTests;
    }

};

#endif
