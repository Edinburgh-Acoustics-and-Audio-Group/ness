/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ScoreParserZero
 */
#ifndef _TESTSCOREPARSERZERO_H_
#define _TESTSCOREPARSERZERO_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ScoreParserZero.h"
#include "DummyComponent2D.h"

class TestScoreParserZero : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserZero();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserZeroTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserZero>("testScoreParserZero", &TestScoreParserZero::testScoreParserZero));
        return suiteOfTests;
    }


 private:
    Instrument *instrument;
    DummyComponent2D *comp1, *comp2;
};

#endif
