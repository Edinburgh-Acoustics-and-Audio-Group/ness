/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InstrumentParserSoundboard
 */
#ifndef _TESTINSTRUMENTPARSERSOUNDBOARD_H_
#define _TESTINSTRUMENTPARSERSOUNDBOARD_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserSoundboard.h"

class TestInstrumentParserSoundboard : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserSoundboard();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserSoundboardTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserSoundboard>("testInstrumentParserSoundboard", &TestInstrumentParserSoundboard::testInstrumentParserSoundboard));
        return suiteOfTests;
    }

};

#endif
