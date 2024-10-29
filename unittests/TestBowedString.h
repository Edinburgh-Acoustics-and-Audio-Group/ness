/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for BowedString
 */
#ifndef _TESTBOWEDSTRING_H_
#define _TESTBOWEDSTRING_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "BowedString.h"

class TestBowedString : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testBowedString();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("BowedStringTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestBowedString>("testBowedString", &TestBowedString::testBowedString));
        return suiteOfTests;
    }
};

#endif
