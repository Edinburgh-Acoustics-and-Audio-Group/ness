/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for PlateEmbedded
 */
#ifndef _TESTPLATEEMBEDDED_H_
#define _TESTPLATEEMBEDDED_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "PlateEmbedded.h"
#include "AirboxIndexed.h"
#include "Embedding.h"

class TestPlateEmbedded : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testPlateEmbedded();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("PlateEmbeddedTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestPlateEmbedded>("testPlateEmbedded", &TestPlateEmbedded::testPlateEmbedded));
        return suiteOfTests;
    }

 private:
    PlateEmbedded *plate;
    AirboxIndexed *airbox;
    Embedding *embedding;

};

#endif
