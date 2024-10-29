/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InstrumentParserBrass
 */
#ifndef _TESTINSTRUMENTPARSERBRASS_H_
#define _TESTINSTRUMENTPARSERBRASS_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserBrass.h"

class TestInstrumentParserBrass : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserBrass();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserBrassTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserBrass>("testInstrumentParserBrass", &TestInstrumentParserBrass::testInstrumentParserBrass));
        return suiteOfTests;
    }
};

#endif
