/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestStringWithFrets.h"

void TestStringWithFrets::setUp()
{
    str = new StringWithFrets("string", 0.7, 0.02, 80.0, 2.1, 0.0003, 10.0, 8.0, 0.2, 0.3, 0.8, 0.3,
			      4, -0.001, -0.002, 0.015);
    str->setParams(1.8e15, 1.555095115459269, 50);
}

void TestStringWithFrets::tearDown()
{
    delete str;
}

void TestStringWithFrets::testStringWithFrets()
{
    // check state size
    CPPUNIT_ASSERT_EQUAL(488, str->getStateSize());

    // put in an impulse
    str->getU1()[244] = 1.0;

    // run a few timesteps
    int i;
    for (i = 0; i < 10; i++) {
	str->runTimestep(i);
	str->swapBuffers(i);
    }

    // FIXME: the results are identical to in the normal string with no frets, suggesting we're not
    // really testing the behaviour fully
    // check result
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.117309096166934, str->getU()[243], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.883398822252997, str->getU()[244], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.117309096166934, str->getU()[245], 1e-12);
}

