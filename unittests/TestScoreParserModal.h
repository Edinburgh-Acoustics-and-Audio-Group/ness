/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Unit tests for ScoreParserModal
 */
#ifndef _TESTSCOREPARSERMODAL_H_
#define _TESTSCOREPARSERMODAL_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ScoreParserModal.h"

class TestScoreParserModal : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserModal();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserModalTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserModal>("testScoreParserModal", &TestScoreParserModal::testScoreParserModal));
        return suiteOfTests;
    }


 private:
};

#endif
