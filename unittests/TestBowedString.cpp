/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestBowedString.h"
#include "GlobalSettings.h"

void TestBowedString::setUp()
{
}

void TestBowedString::tearDown()
{
}

void TestBowedString::testBowedString()
{
    int i;
    GlobalSettings::getInstance()->setSampleRate(44100);

    //============================================================
    //
    // First test basic string on its own
    //
    //============================================================
    BowedString *bs = new BowedString("string1", 440.0, 7e-4, 3e-4, 2e11, 10.0, 8.0, 0.35);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(32, bs->getStateSize());

    // put in an impulse
    bs->getU1()[16] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	bs->runTimestep(i);
	bs->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0000874294864440418, bs->getU1()[16], 1e-10);

    // run some more
    for (; i < 2000; i++) {
	bs->runTimestep(i);
	bs->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0000848908921407052, bs->getU1()[16], 1e-10);

    delete bs;

    //============================================================
    //
    // Test with bow
    //
    //============================================================
    vector<double> bowtime, bowpos, bowforcew, bowforceu;

    bowtime.push_back(0.0);
    bowtime.push_back(0.1);
    bowtime.push_back(0.5);

    bowpos.push_back(0.86);
    bowpos.push_back(0.86);
    bowpos.push_back(0.86);

    bowforcew.push_back(0.0);
    bowforcew.push_back(-0.3);
    bowforcew.push_back(-0.3);

    bowforceu.push_back(0.0);
    bowforceu.push_back(0.0);
    bowforceu.push_back(2.0);

    bs = new BowedString("string1", 440.0, 7e-4, 3e-4, 2e11, 10.0, 8.0, 0.35);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(32, bs->getStateSize());

    // set up the bow
    bs->addBow(1e-8, 0.0, 0.0, 0.0, &bowtime, &bowpos, &bowforcew, &bowforceu);

    // run some timesteps
    for (i = 0; i < 10000; i++) {
	bs->runTimestep(i);
	bs->swapBuffers(i);
    }

    // check value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.59885654511967e-5, bs->getU1()[16], 1e-10);

    // run some more
    for (; i < 20000; i++) {
	bs->runTimestep(i);
	bs->swapBuffers(i);
    }

    // check value again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000108407105708584, bs->getU1()[16], 1e-10);

    delete bs;


    //============================================================
    //
    // Test with bow and finger
    //
    //============================================================
    vector<double> vibrato;

    bs = new BowedString("string1", 440.0, 7e-4, 3e-4, 2e11, 10.0, 8.0, 0.35);
    bs->setBowParameters(1e6, 2.0, 20.0, 10.0, 0.1);
    bs->setFingerParameters(1e5, 1e3, 2.2, 50, 20, 0.05);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(32, bs->getStateSize());

    // set up the bow
    // vector contents same as in previous test case
    bs->addBow(1e-8, 0.0, 0.0, 0.0, &bowtime, &bowpos, &bowforcew, &bowforceu);

    // set up the finger
    bowtime.clear();
    bowpos.clear();
    bowforcew.clear();
    bowforceu.clear();

    bowtime.push_back(0);
    bowtime.push_back(0.1);
    bowtime.push_back(0.5);

    bowpos.push_back(0.159104);
    bowpos.push_back(0.159104);
    bowpos.push_back(0.159104);

    bowforcew.push_back(0);
    bowforcew.push_back(-3);
    bowforcew.push_back(-3);

    bowforceu.push_back(0);
    bowforceu.push_back(0);
    bowforceu.push_back(0);

    vibrato.push_back(0.3);
    vibrato.push_back(1.0);
    vibrato.push_back(0.3);
    vibrato.push_back(0.02);
    vibrato.push_back(5.0);

    bs->addFinger(1e-8, 0.0, 0.0, 0.0, &bowtime, &bowpos, &bowforcew, &bowforceu,
		  &vibrato);

    // run some timesteps
    for (i = 0; i < 10000; i++) {
	bs->runTimestep(i);
	bs->swapBuffers(i);
    }

    // check value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.96857615919453e-5, bs->getU1()[16], 1e-10);

    // run some more
    for (; i < 20000; i++) {
	bs->runTimestep(i);
	bs->swapBuffers(i);
    }

    // check value again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-7.40680450263896e-5, bs->getU1()[16], 1e-10);

    delete bs;
}

