/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for GuitarString
 */
#ifndef _TESTGUITARSTRING_H_
#define _TESTGUITARSTRING_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "GuitarString.h"

class TestGuitarString : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testGuitarString();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("GuitarStringTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestGuitarString>("testGuitarString", &TestGuitarString::testGuitarString));
        return suiteOfTests;
    }
};

#endif
