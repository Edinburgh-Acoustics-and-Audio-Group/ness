/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for StringWithFrets
 */
#ifndef _TESTSTRINGWITHFRETS_H_
#define _TESTSTRINGWITHFRETS_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "StringWithFrets.h"

class TestStringWithFrets : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testStringWithFrets();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("StringWithFretsTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestStringWithFrets>("testStringWithFrets", &TestStringWithFrets::testStringWithFrets));
        return suiteOfTests;
    }

 private:
    StringWithFrets *str;
};

#endif
