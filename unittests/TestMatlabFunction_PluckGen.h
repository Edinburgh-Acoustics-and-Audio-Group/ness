/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Unit tests for MatlabFunction_PluckGen
 */
#ifndef _TESTMATLABFUNCTIONPLUCKGEN_H_
#define _TESTMATLABFUNCTIONPLUCKGEN_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MatlabFunction_PluckGen.h"

class TestMatlabFunction_PluckGen : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMatlabFunction_PluckGen();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MatlabFunction_PluckGenTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestMatlabFunction_PluckGen>("testMatlabFunction_PluckGen", &TestMatlabFunction_PluckGen::testMatlabFunction_PluckGen));
        return suiteOfTests;
    }
};

#endif
