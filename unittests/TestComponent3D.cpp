/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestComponent3D.h"

void TestComponent3D::setUp()
{
    component = new DummyComponent3D("component", 10, 10, 10);
}

void TestComponent3D::tearDown()
{
    delete component;
}

void TestComponent3D::testComponent3D()
{
    // check state size and dimensions are as expected
    CPPUNIT_ASSERT_EQUAL(10, component->getNx());
    CPPUNIT_ASSERT_EQUAL(10, component->getNy());
    CPPUNIT_ASSERT_EQUAL(10, component->getNz());
    CPPUNIT_ASSERT_EQUAL(1000, component->getStateSize());

    // test getIndex
    CPPUNIT_ASSERT_EQUAL(423, component->getIndex(2, 3, 4));

    // test getIndexf
    CPPUNIT_ASSERT_EQUAL(576, component->getIndexf(0.7, 0.6, 0.5));

    // test getInterpolationInfo
    InterpolationInfo interp;
    component->getInterpolationInfo(&interp, 0.51, 0.52, 0.53);
    CPPUNIT_ASSERT_EQUAL((int)INTERPOLATION_TRILINEAR, interp.type);
    CPPUNIT_ASSERT_EQUAL(555, interp.idx);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.504, interp.alpha[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.126, interp.alpha[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.056, interp.alpha[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.014, interp.alpha[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.216, interp.alpha[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.054, interp.alpha[5], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.024, interp.alpha[6], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.006, interp.alpha[7], 1e-12);
}

