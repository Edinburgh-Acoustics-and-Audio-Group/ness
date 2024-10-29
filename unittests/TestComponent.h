/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Component
 */
#ifndef _TESTCOMPONENT_H_
#define _TESTCOMPONENT_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Component.h"

class TestComponent : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testComponent();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ComponentTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestComponent>("testComponent", &TestComponent::testComponent));
        return suiteOfTests;
    }

};

#endif
