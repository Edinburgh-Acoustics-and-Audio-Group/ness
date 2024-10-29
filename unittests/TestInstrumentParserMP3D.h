/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InstrumentParserMP3D
 */
#ifndef _TESTINSTRUMENTPARSERMP3D_H_
#define _TESTINSTRUMENTPARSERMP3D_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserMP3D.h"

class TestInstrumentParserMP3D : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserMP3D();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserMP3DTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserMP3D>("testInstrumentParserMP3D", &TestInstrumentParserMP3D::testInstrumentParserMP3D));
        return suiteOfTests;
    }
};

#endif
