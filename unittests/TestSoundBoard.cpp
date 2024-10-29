/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestSoundBoard.h"

void TestSoundBoard::setUp()
{
    int i;
    strings = new vector<ComponentString*>();

    double y = 0.25;
    for (i = 0; i < 10; i++) {
	strings->push_back(new ComponentString("string", 0.7, 0.02, 80.0, 2e11, 0.0003, 10.0, 8.0, 0.2, y,
					       0.8, y));
	y += 0.05;
    }

    soundBoard = new SoundBoard("soundboard", 0.3, 7850.0, 2e11, 0.001, 0, 0.5, 0.2, 10.0, 9.0, 1, strings);
}

void TestSoundBoard::tearDown()
{
    int i;

    // soundBoard will delete the vector, but not the strings within
    for (i = 0; i < strings->size(); i++) {
	delete strings->at(i);
    }
    delete soundBoard;
}

void TestSoundBoard::testSoundBoard()
{
    // put in an impulse
    soundBoard->getU1()[100] = 1.0;

    // run some iterations
    int i, j;
    for (i = 0; i < 100; i++) {
	for (j = 0; j < strings->size(); j++) {
	    strings->at(j)->runTimestep(i);
	}
	soundBoard->runTimestep(i);

	for (j = 0; j < strings->size(); j++) {
	    strings->at(j)->swapBuffers(i);
	}
	soundBoard->swapBuffers(i);
    }

    // check result
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0127914551495629, soundBoard->getU()[99], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.00039023490234321, soundBoard->getU()[100], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, soundBoard->getU()[101], 1e-10);
}

