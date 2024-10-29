/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Plate
 */
#ifndef _TESTPLATE_H_
#define _TESTPLATE_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Plate.h"

class TestPlate : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testPlate();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("PlateTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestPlate>("testPlate", &TestPlate::testPlate));
        return suiteOfTests;
    }

 private:
    Plate *plate;
};

#endif
