/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Component3D
 */
#ifndef _TESTCOMPONENT3D_H_
#define _TESTCOMPONENT3D_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Component3D.h"
#include "DummyComponent3D.h"

class TestComponent3D : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testComponent3D();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("Component3DTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestComponent3D>("testComponent3D", &TestComponent3D::testComponent3D));
        return suiteOfTests;
    }

 private:
    DummyComponent3D *component;
};

#endif
