/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestGuitarString.h"
#include "GlobalSettings.h"

void TestGuitarString::setUp()
{
}

void TestGuitarString::tearDown()
{
}

void TestGuitarString::testGuitarString()
{
    int i;
    GlobalSettings::getInstance()->setSampleRate(48000);

    //============================================================
    //
    // First test basic string on its own
    //
    //============================================================
    GuitarString *gs = new GuitarString("string1", 0.68, 2e11, 9.5, 0.0002, 7850.0, 15.0, 5.0);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(140, gs->getStateSize());

    // put in an impulse
    gs->getU1()[70] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000453837337511491, gs->getU1()[70], 1e-10);

    // run some more
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000478566475402162, gs->getU1()[70], 1e-10);

    delete gs;

    //============================================================
    //
    // Test with backboard
    //
    //============================================================
    gs = new GuitarString("string1", 0.68, 2e11, 9.5, 0.0002, 7850.0, 15.0, 5.0);

    // set up backboard
    gs->setBarrierParams(1e10, 1.3, 10.0, 20, 1e-12);
    gs->setBackboard(-0.002, -0.001, -0.0002);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(140, gs->getStateSize());

    // put in an impulse
    gs->getU1()[70] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000779325475290985, gs->getU1()[70], 1e-10);

    // run some more
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00158494390381735, gs->getU1()[70], 1e-10);

    delete gs;


    //============================================================
    //
    // Test with frets
    //
    //============================================================
    static const double fp[] = {
	0.056125687318306, 0.109101281859661, 0.159103584746286, 0.206299474015900,
	0.250846461561659, 0.292893218813452, 0.332580072914983, 0.370039475052563,
	0.405396442498639, 0.438768975845314, 0.470268452820352, 0.500000000000000,
	0.528062843659153, 0.554550640929830, 0.579551792373143, 0.603149737007950,
	0.625423230780830, 0.646446609406726, 0.666290036457491, 0.685019737526282,
	-1.0
    };
    vector<double> fretpos, fretheight;

    i = 0;
    while (fp[i] >= 0.0) {
	fretpos.push_back(fp[i]);
	fretheight.push_back(-0.001);
	i++;
    }

    gs = new GuitarString("string1", 0.68, 2e11, 9.5, 0.0002, 7850.0, 15.0, 5.0);

    // set up backboard
    gs->setBarrierParams(1e10, 1.3, 10.0, 20, 1e-12);
    gs->setFrets(fretpos, fretheight);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(140, gs->getStateSize());

    // put in an impulse
    gs->getU1()[70] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000777046005609474, gs->getU1()[70], 1e-10);

    // run some more
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000586025267342512, gs->getU1()[70], 1e-10);

    delete gs;


    //============================================================
    //
    // Test with finger
    //
    //============================================================
    vector<double> fingertime, fingerpos, fingerforce;
    fingertime.push_back(0.0);
    fingerpos.push_back(0.04);
    fingerforce.push_back(0.0);

    fingertime.push_back(0.18);
    fingerpos.push_back(0.15);
    fingerforce.push_back(0.0);

    fingertime.push_back(0.31);
    fingerpos.push_back(0.09);
    fingerforce.push_back(1.0);

    fingertime.push_back(0.78);
    fingerpos.push_back(0.17);
    fingerforce.push_back(1.0);

    fingertime.push_back(1.0);
    fingerpos.push_back(0.28);
    fingerforce.push_back(1.0);

    gs = new GuitarString("string1", 0.68, 2e11, 9.5, 0.0002, 7850.0, 15.0, 5.0);

    // set up fingers
    gs->setFingerParams(0.005, 1e7, 3.3, 100.0);
    gs->addFinger(0.01, 0.0, &fingertime, &fingerpos, &fingerforce);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(140, gs->getStateSize());

    // put in an impulse
    gs->getU1()[70] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000453837337511491, gs->getU1()[70], 1e-10);

    // run some more
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000478566475402162, gs->getU1()[70], 1e-10);

    delete gs;

    //============================================================
    //
    // Test with all three
    //
    //============================================================
    gs = new GuitarString("string1", 0.68, 2e11, 9.5, 0.0002, 7850.0, 15.0, 5.0);

    // set up backboard
    gs->setBarrierParams(1e10, 1.3, 10.0, 20, 1e-12);
    gs->setBackboard(-0.002, -0.001, -0.0002);
    gs->setFrets(fretpos, fretheight);

    // set up fingers
    gs->setFingerParams(0.005, 1e7, 3.3, 100.0);
    gs->addFinger(0.01, 0.0, &fingertime, &fingerpos, &fingerforce);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(140, gs->getStateSize());

    // put in an impulse
    gs->getU1()[70] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000387441962829324, gs->getU1()[70], 1e-10);

    // run some more
    for (i = 0; i < 1000; i++) {
	gs->runTimestep(i);
	gs->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000158731018211085, gs->getU1()[70], 1e-10);

    delete gs;
}

