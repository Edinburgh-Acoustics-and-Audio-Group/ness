/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Component1D
 */
#ifndef _TESTCOMPONENT1D_H_
#define _TESTCOMPONENT1D_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Component1D.h"
#include "DummyComponent1D.h"

class TestComponent1D : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testComponent1D();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("Component1DTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestComponent1D>("testComponent1D", &TestComponent1D::testComponent1D));
        return suiteOfTests;
    }

 private:
    DummyComponent1D *component;

};

#endif
