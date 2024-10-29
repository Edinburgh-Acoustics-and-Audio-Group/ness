/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestComponentString.h"

void TestComponentString::setUp()
{
    str = new ComponentString("string", 0.7, 0.02, 80.0, 2.1, 0.0003, 10.0, 8.0, 0.2, 0.3, 0.8, 0.3);
}

void TestComponentString::tearDown()
{
    delete str;
}

void TestComponentString::testComponentString()
{
    // check state size
    CPPUNIT_ASSERT_EQUAL(488, str->getStateSize());

    // check basic parameters
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, str->getXc1(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.3, str->getYc1(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, str->getXc2(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.3, str->getYc2(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.02, str->getRho(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.38155105579643, str->getSig0(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00143737166324435, str->getH(), 1e-12);

    // put in an impulse
    str->getU1()[244] = 1.0;

    // run a few timesteps
    int i;
    for (i = 0; i < 10; i++) {
	str->runTimestep(i);
	str->swapBuffers(i);
    }

    // check result
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.117309096166934, str->getU()[243], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.883398822252997, str->getU()[244], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.117309096166934, str->getU()[245], 1e-12);

    // run some more timesteps
    for (i = 11; i < 300; i++) {
	str->runTimestep(i);
	str->swapBuffers(i);
    }

    // check connection values
    double l[2];
    l[0] = 0.0;
    l[1] = 0.0;
    str->getConnectionValues(l);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.99699245700649, l[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.990019527692567, l[1], 1e-12);

    // try setting connection values too
    l[0] = 0.1;
    l[1] = 0.2;
    str->setConnectionValues(l);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.985023287363941, str->getU()[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.00003181371914, str->getU()[487], 1e-12);
}

