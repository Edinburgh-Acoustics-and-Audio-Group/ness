/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestBrassInstrument.h"
#include "GlobalSettings.h"
#include "InputValve.h"

#include <vector>
using namespace std;

void TestBrassInstrument::setUp()
{
}

void TestBrassInstrument::tearDown()
{
}

static const double boredata[] = {
    0, 17.34,
    1, 16.3,
    2, 15.28,
    3, 13.8,
    4, 12.3,
    5, 9.59,
    6, 7.38,
    7, 5.53,
    8, 4.95,
    9, 4.46,
    10, 4.16,
    12, 4.08,
    16, 4.08,
    52.3, 5.8,
    57.3, 6.2,
    62.3, 6.5,
    67.3, 6.8,
    72.3, 7.1,
    77.3, 7.5,
    82.3, 7.8,
    87.3, 8.1,
    87.4, 9.7,
    88.3, 9.6,
    88.4, 8.7,
    91.3, 8.7,
    96.3, 8.8,
    98.3, 8.7,
    160.3, 10.1,
    210.3, 10.8,
    260.3, 11.4,
    285.3, 11.7,
    318.3, 11.75,
    361.3, 11.75,
    396.3, 11.75,
    561.3, 11.75,
    636.3, 11.75,
    651.3, 11.75,
    726.3, 11.75,
    741.3, 11.8,
    791.3, 11.8,
    841.3, 11.8,
    941.3, 12.9,
    981.3, 13.8,
    1031.3, 14.9,
    1041.3, 15.3,
    1081.3, 16.5,
    1131.3, 18.3,
    1141.3, 18.8,
    1181.3, 20.5,
    1242.3, 24.5,
    1288.3, 30,
    1303.8, 33.1,
    1317.3, 36.6,
    1327.3, 40.4,
    1335.3, 44.7,
    1343.3, 49.4,
    1348.3, 54.6,
    1354.3, 60.3,
    1358.8, 66.7,
    1363.8, 73.8,
    1367.8, 81.4,
    1371.3, 90,
    1375.8, 99.5,
    1378.3, 109.8,
    1381.3, 127
};

void TestBrassInstrument::testBrassInstrument()
{
    int i;
    vector<double> vpos, vdl, vbl, bore;

    GlobalSettings::getInstance()->setSampleRate(44100);

    // populate vpos, vdl, vbl, bore
    vpos.push_back(600.0);
    vdl.push_back(20.0);
    vbl.push_back(200.0);
    for (i = 0; i < (65*2); i++) {
	bore.push_back(boredata[i]);
    }
    BrassInstrument *brass = new BrassInstrument("brass", 20.0, 1, vpos, vdl, vbl, bore);

    // set valve input
    vector<double>opening, vibratoFrequency, vibratoAmplitude;
    opening.push_back(0.0);
    opening.push_back(1.0);
    vibratoAmplitude.push_back(0.0);
    vibratoAmplitude.push_back(0.0);
    vibratoFrequency.push_back(0.0);
    vibratoFrequency.push_back(100.0);
    InputValve *valve = new InputValve(brass, 0, opening, vibratoFrequency, vibratoAmplitude);

    // check total state size
    CPPUNIT_ASSERT_EQUAL(176, brass->getStateSize());

    // put in an impulse
    brass->getU1()[50] = 0.001;

    // run some timesteps
    for (i = 0; i < 1000; i++) {
	brass->runTimestep(i);
	brass->swapBuffers(i);
    }

    // check the value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.413388413259433, brass->getU1()[50], 1e-10);

    // run some more
    for (; i < 2000; i++) {
	brass->runTimestep(i);
	brass->swapBuffers(i);
    }

    // check again
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0315684666016029, brass->getU1()[50], 1e-10);

    delete valve;
    delete brass;
}
