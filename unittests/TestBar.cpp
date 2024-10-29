/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestBar.h"
#include "MaterialsManager.h"

void TestBar::setUp()
{
    bar = new Bar("bar", MaterialsManager::getInstance()->getMaterial("steel"), 0.2, 0.01, 1);
}

void TestBar::tearDown()
{
    delete bar;
}

void TestBar::testBar()
{
    int i;

    // check state size is as expected
    CPPUNIT_ASSERT_EQUAL(8, bar->getStateSize());

    // put in an impulse
    bar->getU1()[4] = 1.0;

    // run several timesteps
    for (i = 0; i < 10; i++) {
	bar->runTimestep(i);
	bar->swapBuffers(i);
    }

    // now check surrounding values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.39601834409212, bar->getU()[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.70553719500611, bar->getU()[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.886874864493088, bar->getU()[5], 1e-12);
}

