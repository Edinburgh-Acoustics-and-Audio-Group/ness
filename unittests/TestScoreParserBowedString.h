/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for ScoreParserBowedString
 */
#ifndef _TESTSCOREPARSERBOWEDSTRING_H_
#define _TESTSCOREPARSERBOWEDSTRING_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ScoreParserBowedString.h"

class TestScoreParserBowedString : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserBowedString();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserBowedStringTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserBowedString>("testScoreParserBowedString", &TestScoreParserBowedString::testScoreParserBowedString));
        return suiteOfTests;
    }
};

#endif
