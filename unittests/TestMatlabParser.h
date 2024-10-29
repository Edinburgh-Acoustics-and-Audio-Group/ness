/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for MatlabParser
 */
#ifndef _TESTMATLABPARSER_H_
#define _TESTMATLABPARSER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabParser.h"

class TestMatlabParser : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabParser();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabParserTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabParser>("testMatlabParser", &TestMatlabParser::testMatlabParser));
        return suiteOfTests;
    }
};

#endif
