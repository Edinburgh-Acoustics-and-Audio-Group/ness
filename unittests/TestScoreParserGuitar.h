/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for ScoreParserGuitar
 */
#ifndef _TESTSCOREPARSERGUITAR_H_
#define _TESTSCOREPARSERGUITAR_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ScoreParserGuitar.h"

class TestScoreParserGuitar : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserGuitar();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserGuitarTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserGuitar>("testScoreParserGuitar", &TestScoreParserGuitar::testScoreParserGuitar));
        return suiteOfTests;
    }
};

#endif
