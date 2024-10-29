/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInputStrike.h"

#include <cstdio>
using namespace std;

void TestInputStrike::setUp()
{
    component = new DummyComponent1D("component", 10);
    strike = new InputStrike(component, 0.0, 0.0, 0.0, 0, 0.001, 100.0, 0, 0);
}

void TestInputStrike::tearDown()
{
    delete strike;
    delete component;
}

void TestInputStrike::testInputStrike()
{
    double s, s1, s2;
    int i;

    // first timestep should be 0 since our strike starts at 0
    CPPUNIT_ASSERT_EQUAL(0, Input::getFirstInputTimestep());

    s = 0.0;
    s1 = 0.0;
    s2 = 0.0;

    // run first strike timestep
    strike->runTimestep(0, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s, 1e-18);

    // run second timestep
    s = 0.0;
    strike->runTimestep(1, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.508927905953366, s, 1e-10);

    // run third timestep
    s = 0.0;
    strike->runTimestep(2, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.02535131927513, s, 1e-10);

    // run fourth timestep
    s = 0.0;
    strike->runTimestep(3, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.51840023227408, s, 1e-10);

    // run fifth timestep
    s = 0.0;
    strike->runTimestep(4, &s, &s1, &s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.93732335844094, s, 1e-10);

    // s1 and s2 should be unaffected by a strike
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1, 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2, 1e-12);
}

