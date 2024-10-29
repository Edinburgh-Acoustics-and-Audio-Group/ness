/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Unit tests for the Logger
 */

#ifndef _TESTLOGGER_H_
#define _TESTLOGGER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Logger.h"

class TestLogger : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testLogger();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("LoggerTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestLogger>("testLogger",
								  &TestLogger::testLogger));
	return suiteOfTests;
    }
};

#endif
