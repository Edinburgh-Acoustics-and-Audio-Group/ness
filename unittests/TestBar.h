/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Bar
 */
#ifndef _TESTBAR_H_
#define _TESTBAR_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Bar.h"

class TestBar : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testBar();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("BarTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestBar>("testBar", &TestBar::testBar));
        return suiteOfTests;
    }

 private:
    Bar *bar;

};

#endif
