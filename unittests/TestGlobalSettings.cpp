/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */

#include "TestGlobalSettings.h"

void TestGlobalSettings::setUp()
{
}

void TestGlobalSettings::tearDown()
{
}

void TestGlobalSettings::testGlobalSettings()
{
    GlobalSettings *gs = GlobalSettings::getInstance();
    
    // set some values and check that they are returned the same
    gs->setFixPar(1.5);
    gs->setIterinv(10);
    gs->setHighPassOn(true);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, gs->getFixPar(), 1.0e-12);
    CPPUNIT_ASSERT_EQUAL(10, gs->getIterinv());
    CPPUNIT_ASSERT(gs->getHighPassOn());

    // test computation of k
    gs->setSampleRate(96000.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.04166666667e-5, gs->getK(), 1.0e-12);

    // test number of timesteps
    gs->setDuration(2.0);
    CPPUNIT_ASSERT_EQUAL(192000, gs->getNumTimesteps());

    // set back to defaults
    gs->setSampleRate(44100.0);
    gs->setDuration(1.0);
    gs->setFixPar(1.0);
    gs->setIterinv(5);
    gs->setHighPassOn(false);
}
