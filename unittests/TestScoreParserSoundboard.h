/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ScoreParserSoundboard
 */
#ifndef _TESTSCOREPARSERSOUNDBOARD_H_
#define _TESTSCOREPARSERSOUNDBOARD_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "DummyComponent1D.h"
#include "ScoreParserSoundboard.h"

class TestScoreParserSoundboard : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserSoundboard();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserSoundboardTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserSoundboard>("testScoreParserSoundboard", &TestScoreParserSoundboard::testScoreParserSoundboard));
        return suiteOfTests;
    }

 private:
    Instrument *instrument;
    DummyComponent1D *comp1, *comp2;
};

#endif
