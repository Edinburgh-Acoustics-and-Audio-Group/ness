/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Unit tests for InstrumentParserModal
 */
#ifndef _TESTINSTRUMENTPARSERMODAL_H_
#define _TESTINSTRUMENTPARSERMODAL_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserModal.h"

class TestInstrumentParserModal : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserModal();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserModalTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserModal>("testInstrumentParserModal", &TestInstrumentParserModal::testInstrumentParserModal));
        return suiteOfTests;
    }
};

#endif
