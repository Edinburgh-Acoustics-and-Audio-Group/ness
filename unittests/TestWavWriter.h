/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for WavWriter
 */
#ifndef _TESTWAVWRITER_H_
#define _TESTWAVWRITER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "WavWriter.h"

class TestWavWriter : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testWavWriter();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("WavWriterTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestWavWriter>("testWavWriter", &TestWavWriter::testWavWriter));
        return suiteOfTests;
    }

};

#endif
