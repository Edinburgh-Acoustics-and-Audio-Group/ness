/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for WavReader
 */
#ifndef _TESTWAVREADER_H_
#define _TESTWAVREADER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "WavReader.h"

class TestWavReader : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testWavReader();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("WavReaderTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestWavReader>("testWavReader", &TestWavReader::testWavReader));
        return suiteOfTests;
    }

};

#endif
