/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestInputLips.h"
#include "BrassInstrument.h"
#include "GlobalSettings.h"

void TestInputLips::setUp()
{
}

void TestInputLips::tearDown()
{
}

void TestInputLips::testInputLips()
{
    int i;
    vector<double> vpos, vdl, vbl, bore;

    GlobalSettings::getInstance()->setSampleRate(44100);

    // create a brass instrument to test with
    vpos.push_back(600.0);
    vdl.push_back(20.0);
    vbl.push_back(200.0);
    bore.push_back(0.0);
    bore.push_back(17.34);
    bore.push_back(1381.3);
    bore.push_back(127.0);
    BrassInstrument *brass = new BrassInstrument("brass", 20.0, 1, vpos, vdl, vbl, bore);

    // now create lips input
    vector<double> Sr, mu, sigma, H, w, pressure, lip_frequency, vibamp, vibfreq, tremamp, tremfreq, noiseamp;
    Sr.push_back(0.0);
    Sr.push_back(1.46e-5);
    mu.push_back(0.0);
    mu.push_back(5.37e-5);
    sigma.push_back(0.0);
    sigma.push_back(5.0);
    H.push_back(0.0);
    H.push_back(0.00029);
    w.push_back(0.0);
    w.push_back(0.01);
    pressure.push_back(0.0);
    pressure.push_back(0.0);
    pressure.push_back(10e-3);
    pressure.push_back(3e3);
    lip_frequency.push_back(0.0);
    lip_frequency.push_back(35.0);
    vibamp.push_back(0.0);
    vibamp.push_back(0.01);
    vibfreq.push_back(0.0);
    vibfreq.push_back(1000.0);
    tremamp.push_back(0.0);
    tremamp.push_back(0.001);
    tremfreq.push_back(0.0);
    tremfreq.push_back(44099.0);
    noiseamp.push_back(0.0);
    noiseamp.push_back(0.0);
    InputLips *lips = new InputLips(brass, Sr, mu, sigma, H, w, pressure, lip_frequency, vibamp,
				    vibfreq, tremamp, tremfreq, noiseamp);

    // run some timesteps
    for (i = 0; i < 100; i++) {
	lips->runTimestep(i, brass->getU(), brass->getU1(), brass->getVmainHist());
	brass->swapBuffers(i);
    }

    // check a value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(547.539581282329, brass->getU()[0], 1e-8);

    delete lips;
    delete brass;
}

