/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInputPluck.h"

void TestInputPluck::setUp()
{
    component = new DummyComponent1D("component", 10);
    pluck = new InputPluck(component, 0.0, 0.0, 0.0, 0, 0.001, 100.0, 0, 0);
}

void TestInputPluck::tearDown()
{
    delete pluck;
    delete component;
}

void TestInputPluck::testInputPluck()
{
    double s, s1, s2;

    // first timestep should be 0 since our pluck starts at 0
    CPPUNIT_ASSERT_EQUAL(0, Input::getFirstInputTimestep());

    s = 0.0;
    s1 = 0.0;
    s2 = 0.0;

    // run first pluck timestep
    pluck->runTimestep(0, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s, 1e-18);

    // run second timestep
    s = 0.0;
    pluck->runTimestep(1, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.12179748700879, s, 1e-10);

    // run third timestep
    s = 0.0;
    pluck->runTimestep(2, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.486596562921482, s, 1e-10);

    // run fourth timestep
    s = 0.0;
    pluck->runTimestep(3, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.09261996330972, s, 1e-10);

    // run fifth timestep
    s = 0.0;
    pluck->runTimestep(4, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.93691520308406, s, 1e-10);

    // s1 and s2 should be unaffected by a pluck
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1, 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2, 1e-12);    
}

