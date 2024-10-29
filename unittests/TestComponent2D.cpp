/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestComponent2D.h"

void TestComponent2D::setUp()
{
    component = new DummyComponent2D("component", 10, 10);
}

void TestComponent2D::tearDown()
{
    delete component;
}

void TestComponent2D::testComponent2D()
{
    // check state size and dimensions are as expected
    CPPUNIT_ASSERT_EQUAL(10, component->getNx());
    CPPUNIT_ASSERT_EQUAL(10, component->getNy());
    CPPUNIT_ASSERT_EQUAL(100, component->getStateSize());

    // test getIndex
    CPPUNIT_ASSERT_EQUAL(33, component->getIndex(3, 3));

    // test getIndexf
    CPPUNIT_ASSERT_EQUAL(55, component->getIndexf(0.5, 0.5));

    // test getInterpolationInfo
    InterpolationInfo interp;
    component->getInterpolationInfo(&interp, 0.62, 0.63);
    CPPUNIT_ASSERT_EQUAL((int)INTERPOLATION_BILINEAR, interp.type);
    CPPUNIT_ASSERT_EQUAL(66, interp.idx);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.56, interp.alpha[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.24, interp.alpha[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.14, interp.alpha[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.06, interp.alpha[3], 1e-12);
}

