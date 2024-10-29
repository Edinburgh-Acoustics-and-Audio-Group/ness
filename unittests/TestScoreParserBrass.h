/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ScoreParserBrass
 */
#ifndef _TESTSCOREPARSERBRASS_H_
#define _TESTSCOREPARSERBRASS_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ScoreParserBrass.h"

class TestScoreParserBrass : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserBrass();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserBrassTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserBrass>("testScoreParserBrass", &TestScoreParserBrass::testScoreParserBrass));
        return suiteOfTests;
    }
};

#endif
