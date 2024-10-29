/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Parser
 */
#ifndef _TESTPARSER_H_
#define _TESTPARSER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Parser.h"
#include "DummyParser.h"

class TestParser : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testParser();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ParserTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestParser>("testParser", &TestParser::testParser));
        return suiteOfTests;
    }

 private:
    DummyParser *parser;
};

#endif
