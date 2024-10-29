/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestFretboard.h"
#include "GlobalSettings.h"

#include <vector>
using namespace std;

void TestFretboard::setUp()
{
}

void TestFretboard::tearDown()
{
}

void TestFretboard::testFretboard()
{
    int i;

    GlobalSettings::getInstance()->setSampleRate(44100);
    GlobalSettings::getInstance()->setDuration(1.0);

    Fretboard *fb = new Fretboard("fretboard", 0.68, 2e11, 12.1, 0.0002, 7850, 15, 5, 24,
				  -0.002, -0.0005, -0.001, 1e10, 1.3, 10.0, 15);

    // check state size
    CPPUNIT_ASSERT_EQUAL(131, fb->getStateSize());

    // check alpha value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000101181065524917, fb->getAlpha(), 1e-12);

    // put in an impulse
    fb->getU1()[50] = 0.001;

    // run some timesteps and check state
    for (i = 0; i < 1000; i++) {
	fb->runTimestep(i);
	fb->swapBuffers(i);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000273015999451252, fb->getU1()[50], 1e-12);

    for (; i < 2000; i++) {
	fb->runTimestep(i);
	fb->swapBuffers(i);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000507666501937193, fb->getU1()[50], 1e-12);
    delete fb;

    // create a fresh fretboard
    fb = new Fretboard("fretboard", 0.68, 2e11, 12.1, 0.0002, 7850, 15, 5, 24,
		       -0.002, -0.0005, -0.001, 1e10, 1.3, 10.0, 15);

    // set finger parameters this time
    vector<double> fingertime, fingerpos, fingerforce;

    fingertime.push_back(0.0);
    fingertime.push_back(0.05);
    fingertime.push_back(0.1);
    fingertime.push_back(3.0);

    fingerpos.push_back(0.05);
    fingerpos.push_back(0.05);
    fingerpos.push_back(0.05);
    fingerpos.push_back(0.5);

    fingerforce.push_back(0.0);
    fingerforce.push_back(1.0);
    fingerforce.push_back(1.0);
    fingerforce.push_back(1.0);

    fb->setFingerParams(0.005, 1e8, 2.0, 1.0, 0.01, 0.0, &fingertime, &fingerpos, &fingerforce);

    // put in an impulse
    fb->getU1()[50] = 0.001;

    // run some timesteps and check state
    for (i = 0; i < 1000; i++) {
	fb->runTimestep(i);
	fb->swapBuffers(i);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000273015999451252, fb->getU1()[50], 1e-12);

    for (; i < 2000; i++) {
	fb->runTimestep(i);
	fb->swapBuffers(i);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000468327653173156, fb->getU1()[50], 1e-12);

    delete fb;
}
