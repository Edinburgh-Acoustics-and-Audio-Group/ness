/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for fretboard
 */
#ifndef _TESTFRETBOARD_H_
#define _TESTFRETBOARD_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Fretboard.h"

class TestFretboard : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testFretboard();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("FretboardTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestFretboard>("testFretboard", &TestFretboard::testFretboard));
        return suiteOfTests;
    }

};

#endif
