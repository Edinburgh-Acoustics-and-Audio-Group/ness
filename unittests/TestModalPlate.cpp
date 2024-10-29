/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestModalPlate.h"
#include "GlobalSettings.h"
#include "GPUUtil.h"

#include <cstdio>
#include <cmath>
using namespace std;

void TestModalPlate::setUp()
{
}

void TestModalPlate::tearDown()
{
}

void TestModalPlate::testModalPlate()
{
    // create a modal plate
    ModalPlate *mp = new ModalPlate("modalplate", 9, 0.22, 0.21, 0.0005, 0.3,
				    2e11, 7860.0, 20000.0, 1.0, 0.1);

    // check some basic properties
    CPPUNIT_ASSERT_EQUAL(200, mp->getStateSize());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.22, mp->getLx(), 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.21, mp->getLy(), 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5e-4, mp->getH(), 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7860.0, mp->getRho(), 1e-10);
    
    // check some values in Omega and Cnorm
    double *ov = mp->getOmega();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(326.452453518111, ov[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ov[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ov[2], 1e-10);

    double *Cnorm = mp->getCnorm();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2385832050.0, Cnorm[0], 1e-3);

    // put in impulse (in modal form) and run some timesteps
    int i;
    for (i = 0; i < 200; i++) {
	double P = sin(0.5 * M_PI * ov[(i*3)+1]) * sin(0.5 * M_PI * ov[(i*3)+2]);
	P = ((P / 7860.0) / (0.25*0.21/4.0)) / 8.5e-4;
	P = P / Cnorm[i];
	mp->getU1()[i] = 12300.0 * P;
	mp->getU2()[i] = 12300.0 * P;
    }
    int t;
    for (t = 0; t < 100; t++) {
	mp->runTimestep(t);
	mp->swapBuffers(t);
    }

    // check result
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-8.94764467403781e-23, mp->getU1()[60], 1e-30);

    // run some more
    for (; t < 200; t++) {
	mp->runTimestep(t);
	mp->swapBuffers(t);
    }

    // check result
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.80422534494172e-24, mp->getU1()[60], 1e-30);
    delete mp;

#ifdef USE_GPU
    // create a fresh modal plate
    mp = new ModalPlate("modalplate", 9, 0.22, 0.21, 0.0005, 0.3,
			2e11, 7860.0, 20000.0, 1.0, 0.1);

    // move it to GPU
    CPPUNIT_ASSERT(mp->moveToGPU());

    // put in impulse
    ov = mp->getOmega();
    Cnorm = mp->getCnorm();
    double q1[200];
    for (i = 0; i < 200; i++) {
	double P = sin(0.5 * M_PI * ov[(i*3)+1]) * sin(0.5 * M_PI * ov[(i*3)+2]);
	P = ((P / 7860.0) / (0.25*0.21/4.0)) / 8.5e-4;
	P = P / Cnorm[i];
	q1[i] = 12300.0 * P;
    }
    cudaMemcpyH2D(mp->getU1(), &q1[0], 200*sizeof(double));
    cudaMemcpyH2D(mp->getU2(), &q1[0], 200*sizeof(double));

    for (t = 0; t < 100; t++) {
	mp->runTimestep(t);
	mp->swapBuffers(t);
    }

    // check result
    double val;
    cudaMemcpyD2H(&val, &mp->getU1()[60], sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-8.94764467403781e-23, val, 1e-30);

    // run some more
    for (; t < 200; t++) {
	mp->runTimestep(t);
	mp->swapBuffers(t);
    }

    // check result
    cudaMemcpyD2H(&val, &mp->getU1()[60], sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.80422534494172e-24, val, 1e-30);
    delete mp;
#endif    
}

