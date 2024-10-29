/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestEmbedding.h"

#include "AirboxIndexed.h"
#include "PlateEmbedded.h"
#include "Embedding.h"
#include "GlobalSettings.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif


void TestEmbedding::setUp()
{
}

void TestEmbedding::tearDown()
{
}

void TestEmbedding::testEmbedding()
{
    int i;
    AirboxIndexed *airbox = new AirboxIndexed("airbox", 1.2, 1.2, 1.2, 340.0, 1.21);
    GlobalSettings::getInstance()->setLinear(true);
    PlateEmbedded *plate = new PlateEmbedded("plate", 0.33, 7800.0, 8e11, 0.002, 0.0, 0.8, 0.8,
			      4.0, 0.001, 0.0, 0.0, 0.0);
    Embedding *embedding = new Embedding(airbox, plate);
    
    // put in an impulse on the plate
    plate->getU1()[528] = 1.0;

    // run some timesteps
    for (i = 0; i < 100; i++) {
	airbox->runTimestep(i);
	plate->runTimestep(i);
	embedding->runTimestep(i);

	airbox->swapBuffers(i);
	plate->swapBuffers(i);
    }

    // check that the plate impulse is correctly propagated to the airbox
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.8937053470637, airbox->getU1()[500000], 1e-10);

    delete embedding;
    delete plate;
    delete airbox;

#ifdef USE_GPU
    // test embedding with airbox on GPU and plate on CPU
    airbox = new AirboxIndexed("airbox", 1.2, 1.2, 1.2, 340.0, 1.21);
    plate = new PlateEmbedded("plate", 0.33, 7800.0, 8e11, 0.002, 0.0, 0.8, 0.8,
			      4.0, 0.001, 0.0, 0.0, 0.0);
    embedding = new Embedding(airbox, plate);
    
    airbox->moveToGPU();
    embedding->maybeMoveToGPU();

    // put in an impulse on the plate
    plate->getU1()[528] = 1.0;

    // run some timesteps
    for (i = 0; i < 100; i++) {
	airbox->runTimestep(i);
	plate->runTimestep(i);
	embedding->runTimestep(i);

	airbox->swapBuffers(i);
	plate->swapBuffers(i);
    }

    // check that the plate impulse is correctly propagated to the airbox
    double val;
    cudaMemcpyD2H(&val, &airbox->getU1()[500000], sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.8937053470637, val, 1e-10);

    delete embedding;
    delete plate;
    delete airbox;

#endif
}

