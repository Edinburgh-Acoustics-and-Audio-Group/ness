/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestComponent1D.h"

void TestComponent1D::setUp()
{
    component = new DummyComponent1D("component1D", 20);
}

void TestComponent1D::tearDown()
{
    delete component;
}

void TestComponent1D::testComponent1D()
{
    // check state size is as expected
    CPPUNIT_ASSERT_EQUAL(20, component->getStateSize());

    // test getIndex
    CPPUNIT_ASSERT_EQUAL(4, component->getIndex(4));

    // test getIndexf
    CPPUNIT_ASSERT_EQUAL(8, component->getIndexf(0.4));

    // test getInterpolationInfo
    InterpolationInfo interp;
    component->getInterpolationInfo(&interp, 0.025);
    CPPUNIT_ASSERT_EQUAL((int)INTERPOLATION_LINEAR, interp.type);
    CPPUNIT_ASSERT_EQUAL(0, interp.idx);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, interp.alpha[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, interp.alpha[1], 1e-12);
}

