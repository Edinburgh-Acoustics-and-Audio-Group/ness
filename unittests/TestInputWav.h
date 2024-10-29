/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InputWav
 */
#ifndef _TESTINPUTWAV_H_
#define _TESTINPUTWAV_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InputWav.h"

class TestInputWav : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInputWav();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InputWavTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInputWav>("testInputWav", &TestInputWav::testInputWav));
        return suiteOfTests;
    }

};

#endif
