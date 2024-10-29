/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestAirboxIndexed.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

void TestAirboxIndexed::setUp()
{
    airbox = new AirboxIndexed("airbox", 1.2, 1.2, 1.2, 340.0, 1.21);
}

void TestAirboxIndexed::tearDown()
{
    delete airbox;
}

void TestAirboxIndexed::testAirboxIndexed()
{
    int i;

    // test basic scalars first
    // check the values that we passed in
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2, airbox->getLX(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2, airbox->getLY(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2, airbox->getLZ(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.21, airbox->getRhoA(), 1.0e-12);
    
    // check computed values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0134831460674157, airbox->getQ(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.571806500377929, airbox->getGamma(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, airbox->getLambda(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.26757369614512e-5, airbox->getK(), 1.0e-12);

    // check state array size
    CPPUNIT_ASSERT_EQUAL(90, airbox->getNx());
    CPPUNIT_ASSERT_EQUAL(90, airbox->getNy());
    CPPUNIT_ASSERT_EQUAL(90, airbox->getNz());


    // now put an impulse into the state array
    airbox->getU1()[368595] = 1.0;

    // run a number of timesteps
    for (i = 0; i < 10; i++) {
	airbox->runTimestep(i);
	airbox->swapBuffers(i);
    }

    // check the surrounding values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00408886844944792, airbox->getU()[368594], 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0458197433293799, airbox->getU()[368595], 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00408886844944792, airbox->getU()[368596], 1.0e-12);


    // create a fresh airbox
    delete airbox;
    airbox = new AirboxIndexed("airbox", 1.2, 1.2, 1.2, 340.0, 1.21);

    // add a cylindrical drum shell
    airbox->addDrumShell(30, 60, 0.2);

    // put an impulse into the state array
    airbox->getU1()[368595] = 1.0;

    // run more timesteps so the wave has time to bounce off the shell
    for (i = 0; i < 100; i++) {
	airbox->runTimestep(i);
	airbox->swapBuffers(i);
    }

    // check the surrounding values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00885841093518801, airbox->getU()[368594], 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00970786556487715, airbox->getU()[368595], 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00495716938975464, airbox->getU()[368596], 1.0e-12);

#ifdef USE_GPU
    // create a fresh airbox
    delete airbox;
    airbox = new AirboxIndexed("airbox", 1.2, 1.2, 1.2, 340.0, 1.21);

    // move it to the GPU
    CPPUNIT_ASSERT(airbox->moveToGPU());

    // put an impulse into the state array
    double one = 1.0;
    cudaMemcpyH2D(&airbox->getU1()[368595], &one, sizeof(double));

    // run a number of timesteps
    for (i = 0; i < 10; i++) {
	airbox->runTimestep(i);
	airbox->swapBuffers(i);
    }

    // check the surrounding values
    double gpuResult[3];
    cudaMemcpyD2H(&gpuResult[0], &airbox->getU()[368594], 3 * sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00408886844944792, gpuResult[0], 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0458197433293799, gpuResult[1], 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00408886844944792, gpuResult[2], 1.0e-12);

#endif
}

