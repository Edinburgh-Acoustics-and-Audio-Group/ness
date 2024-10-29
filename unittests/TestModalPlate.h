/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Unit tests for ModalPlate
 */
#ifndef _TESTMODALPLATE_H_
#define _TESTMODALPLATE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ModalPlate.h"

class TestModalPlate : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testModalPlate();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ModalPlateTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestModalPlate>("testModalPlate", &TestModalPlate::testModalPlate));
        return suiteOfTests;
    }
};

#endif
