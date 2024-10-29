/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InstrumentParserZeroPt1
 */
#ifndef _TESTINSTRUMENTPARSERZEROPT1_H_
#define _TESTINSTRUMENTPARSERZEROPT1_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserZeroPt1.h"

class TestInstrumentParserZeroPt1 : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserZeroPt1();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserZeroPt1Test");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserZeroPt1>("testInstrumentParserZeroPt1", &TestInstrumentParserZeroPt1::testInstrumentParserZeroPt1));
        return suiteOfTests;
    }

};

#endif
