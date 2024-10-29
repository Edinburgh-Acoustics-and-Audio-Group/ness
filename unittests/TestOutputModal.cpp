/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestOutputModal.h"
#include "ModalPlate.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

#include <cmath>

void TestOutputModal::setUp()
{
}

void TestOutputModal::tearDown()
{
}

void TestOutputModal::testOutputModal()
{
    // create a modal plate
    ModalPlate *mp = new ModalPlate("modalplate", 9, 0.22, 0.21, 0.0005, 0.3,
				    2e11, 7860.0, 20000.0, 1.0, 0.1);

    // set state array contents
    double *u = mp->getU();
    int i, ss;
    ss = mp->getStateSize();
    for (i = 0; i < ss; i++) {
	u[i] = 0.1 * sin(((double)i)/((double)ss) * M_PI * 2.0);
    }

    // create modal output
    OutputModal *om = new OutputModal(mp, 0.5, 0.3, 0.4);

    // check value
    om->runTimestep(0);
    double *data = om->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-390.863563289937, data[0], 1e-10);

    // change state array
    for (i = 0; i < ss; i++) {
	u[i] = 0.08 * cos(((double)i)/((double)ss) * M_PI * 2.0);
    }

    // check value again
    om->runTimestep(1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(374.796858279353, data[1], 1e-10);

    delete om;
    delete mp;

    // other methods are inherited from base output class so they'll be
    // tested by TestOutput

#ifdef USE_GPU
    // create a fresh modal plate
    mp = new ModalPlate("modalplate", 9, 0.22, 0.21, 0.0005, 0.3,
			2e11, 7860.0, 20000.0, 1.0, 0.1);

    // create a fresh modal output
    om = new OutputModal(mp, 0.5, 0.3, 0.4);

    // move them to the GPU
    mp->moveToGPU();
    om->maybeMoveToGPU();

    // initialise state array
    double *h_u = new double[ss];
    ss = mp->getStateSize();
    u = mp->getU();
    for (i = 0; i < ss; i++) {
	h_u[i] = 0.1 * sin(((double)i)/((double)ss) * M_PI * 2.0);
    }
    cudaMemcpyH2D(u, h_u, ss * sizeof(double));

    // get first output value
    om->runTimestep(0);

    // re-initialise state array
    for (i = 0; i < ss; i++) {
	h_u[i] = 0.08 * cos(((double)i)/((double)ss) * M_PI * 2.0);
    }
    cudaMemcpyH2D(u, h_u, ss * sizeof(double));

    // get second output value
    om->runTimestep(1);

    om->copyFromGPU();
    data = om->getData();

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-390.863563289937, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(374.796858279353, data[1], 1e-10);

    delete[] h_u;
    delete om;
    delete mp;
#endif
}

