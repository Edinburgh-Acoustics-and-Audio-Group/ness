/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Unit tests for OutputModal
 */
#ifndef _TESTOUTPUTMODAL_H_
#define _TESTOUTPUTMODAL_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "OutputModal.h"

class TestOutputModal : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testOutputModal();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("OutputModalTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestOutputModal>("testOutputModal", &TestOutputModal::testOutputModal));
        return suiteOfTests;
    }
};

#endif
