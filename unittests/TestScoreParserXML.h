/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ScoreParserXML
 */
#ifndef _TESTSCOREPARSERXML_H_
#define _TESTSCOREPARSERXML_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ScoreParserXML.h"
#include "DummyComponent2D.h"

class TestScoreParserXML : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testScoreParserXML();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ScoreParserXMLTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestScoreParserXML>("testScoreParserXML", &TestScoreParserXML::testScoreParserXML));
        return suiteOfTests;
    }


 private:
    Instrument *instrument;
    DummyComponent2D *comp1, *comp2;
};

#endif
