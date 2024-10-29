/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for InstrumentParserGuitar
 */
#ifndef _TESTINSTRUMENTPARSERGUITAR_H_
#define _TESTINSTRUMENTPARSERGUITAR_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserGuitar.h"

class TestInstrumentParserGuitar : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserGuitar();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserGuitarTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserGuitar>("testInstrumentParserGuitar", &TestInstrumentParserGuitar::testInstrumentParserGuitar));
        return suiteOfTests;
    }
};

#endif
