/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for InstrumentParserBowedString
 */
#ifndef _TESTINSTRUMENTPARSERBOWEDSTRING_H_
#define _TESTINSTRUMENTPARSERBOWEDSTRING_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserBowedString.h"

class TestInstrumentParserBowedString : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserBowedString();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserBowedStringTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserBowedString>("testInstrumentParserBowedString", &TestInstrumentParserBowedString::testInstrumentParserBowedString));
        return suiteOfTests;
    }
};

#endif
