/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestPlate.h"
#include "Material.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

void TestPlate::setUp()
{
    Material *mat = new Material("steel", 2e11, 0.3, 7850.0);
    plate = new Plate("plate1", mat, 0.002, 10.0, 0.3, 0.2, 10.0, 6.0, 4);
    delete mat;
}

void TestPlate::tearDown()
{
    delete plate;
}

void TestPlate::testPlate()
{
    // check basic properties are correct
    CPPUNIT_ASSERT_EQUAL(247, plate->getStateSize());

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.178995274e-7, plate->getAlpha(), 1.0e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0408150478895, plate->getBowFactor(), 1.0e-10);

    plate->getU1()[100] = 1.0;
    int t;
    for (t = 0; t < 1000; t++) {
	plate->runTimestep(t);
	plate->swapBuffers(t);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.42676028225, plate->getU1()[100], 1.0e-10);

    for (t = 1000; t < 2000; t++) {
	plate->runTimestep(t);
	plate->swapBuffers(t);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.301164168295, plate->getU1()[100], 1.0e-10);

#ifdef USE_GPU
    // create a fresh plate
    delete plate;
    Material *mat = new Material("steel", 2e11, 0.3, 7850.0);
    plate = new Plate("plate1", mat, 0.002, 10.0, 0.3, 0.2, 10.0, 6.0, 4);
    delete mat;

    // move it to GPU
    CPPUNIT_ASSERT(plate->moveToGPU());

    // put in an impulse
    double val = 1.0;
    cudaMemcpyH2D(&plate->getU1()[100], &val, sizeof(double));

    for (t = 0; t < 1000; t++) {
	plate->runTimestep(t);
	plate->swapBuffers(t);
    }

    cudaMemcpyD2H(&val, &plate->getU1()[100], sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.42676028225, val, 1.0e-10);

    for (t = 1000; t < 2000; t++) {
	plate->runTimestep(t);
	plate->swapBuffers(t);
    }

    cudaMemcpyD2H(&val, &plate->getU1()[100], sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.301164168295, val, 1.0e-10);
#endif
}

