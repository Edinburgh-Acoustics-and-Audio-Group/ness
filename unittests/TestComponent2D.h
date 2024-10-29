/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Component2D
 */
#ifndef _TESTCOMPONENT2D_H_
#define _TESTCOMPONENT2D_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Component2D.h"
#include "DummyComponent2D.h"

class TestComponent2D : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testComponent2D();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("Component2DTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestComponent2D>("testComponent2D", &TestComponent2D::testComponent2D));
        return suiteOfTests;
    }

 private:
    DummyComponent2D *component;
};

#endif
