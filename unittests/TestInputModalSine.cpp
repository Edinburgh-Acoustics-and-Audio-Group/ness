/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestInputModalSine.h"
#include "ModalPlate.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

void TestInputModalSine::setUp()
{
}

void TestInputModalSine::tearDown()
{
}

void TestInputModalSine::testInputModalSine()
{
    // create a modal plate
    ModalPlate *mp = new ModalPlate("modalplate", 9, 0.22, 0.21, 0.0005, 0.3,
				    2e11, 7860.0, 20000.0, 1.0, 0.1);

    // create a modal sine
    InputModalSine *ms = new InputModalSine(mp, 0.0, 0.5, 369.112060210515202, 4.0,
					    2.0, 2.0, 0.38, 0.37);

    // first timestep should be zero since our sine starts at 0
    CPPUNIT_ASSERT_EQUAL(0, Input::getFirstInputTimestep());

    double *s = mp->getU();
    double *s1 = mp->getU1();
    double *s2 = mp->getU2();

    // run first sine timestep
    ms->runTimestep(0, s, s1, s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[3], 1e-10);

    // run second sine timestep
    ms->runTimestep(1, s, s1, s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.61377533157483e-15, s[0], 1e-25);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.24278983959125e-15, s[1], 1e-25);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.38999542644913e-15, s[2], 1e-25);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.58500726752038e-15, s[3], 1e-25);

    // run third sine timestep
    ms->runTimestep(2, s, s1, s2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.80664149569031e-14, s[0], 1e-24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.12124214112458e-14, s[1], 1e-24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.19483490692029e-14, s[2], 1e-24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.92395663187265e-15, s[3], 1e-25);

    // s1 and s2 should be unaffected by a sine
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[3], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[3], 1e-10);

    delete ms;
    delete mp;

#ifdef USE_GPU
    // create a fresh modal plate
    mp = new ModalPlate("modalplate", 9, 0.22, 0.21, 0.0005, 0.3,
			2e11, 7860.0, 20000.0, 1.0, 0.1);
    int ss = mp->getStateSize();
    s = mp->getU();
    s1 = mp->getU1();
    s2 = mp->getU2();

    // create a fresh modal sine
    ms = new InputModalSine(mp, 0.0, 0.5, 369.112060210515202, 4.0,
			    2.0, 2.0, 0.38, 0.37);

    // move it to the GPU
    ms->moveToGPU();

    // allocate a fake state buffer on the GPU
    double *d_u, *d_u1, *d_u2;
    d_u = (double *)cudaMallocNess(ss * sizeof(double));
    d_u1 = (double *)cudaMallocNess(ss * sizeof(double));
    d_u2 = (double *)cudaMallocNess(ss * sizeof(double));
    
    cudaMemsetNess(d_u, 0, ss * sizeof(double));
    cudaMemsetNess(d_u1, 0, ss * sizeof(double));
    cudaMemsetNess(d_u2, 0, ss * sizeof(double));

    // run timestep
    ms->runTimestep(0, d_u, d_u1, d_u2);
    cudaMemcpyD2H(s, d_u, 4 * sizeof(double));    
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s[3], 1e-10);

    // run second strike timestep
    ms->runTimestep(1, d_u, d_u1, d_u2);
    cudaMemcpyD2H(s, d_u, 4 * sizeof(double));    
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.61377533157483e-15, s[0], 1e-25);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.24278983959125e-15, s[1], 1e-25);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.38999542644913e-15, s[2], 1e-25);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.58500726752038e-15, s[3], 1e-25);

    // run third strike timestep
    ms->runTimestep(2, d_u, d_u1, d_u2);
    cudaMemcpyD2H(s, d_u, 4 * sizeof(double));    
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.80664149569031e-14, s[0], 1e-24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.12124214112458e-14, s[1], 1e-24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.19483490692029e-14, s[2], 1e-24);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.92395663187265e-15, s[3], 1e-25);

    // s1 and s2 should be unaffected by a strike
    cudaMemcpyD2H(s1, d_u1, 4 * sizeof(double));    
    cudaMemcpyD2H(s2, d_u2, 4 * sizeof(double));    
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s1[3], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s2[3], 1e-10);

    cudaFreeNess(d_u);
    cudaFreeNess(d_u1);
    cudaFreeNess(d_u2);

    delete ms;
    delete mp;
#endif
}

