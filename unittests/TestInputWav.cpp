/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInputWav.h"

#include "DummyComponent1D.h"
#include "InputWav.h"

void TestInputWav::setUp()
{
}

void TestInputWav::tearDown()
{
}

void TestInputWav::testInputWav()
{
    DummyComponent1D *comp = new DummyComponent1D("comp", 10);
    InputWav *wav = new InputWav(comp, 0.0, 0.0, 0.0, "sine16m.wav", 0.0,
				 1.0, 0, 0);
    double u = 0.0, u1 = 0.0, u2 = 0.0;

    wav->runTimestep(0, &u, &u1, &u2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.81373708096564e-5, u, 1e-10);
    wav->runTimestep(1, &u, &u1, &u2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0625834254986461, u, 1e-10);
    wav->runTimestep(2, &u, &u1, &u2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.187674001754319, u, 1e-10);
    wav->runTimestep(3, &u, &u1, &u2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.374547118721635, u, 1e-10);
    wav->runTimestep(4, &u, &u1, &u2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.62263071583845, u, 1e-10);

    delete wav;
    delete comp;
}

