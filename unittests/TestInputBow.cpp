/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestInputBow.h"

#include "InputBow.h"
#include "DummyComponent1D.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

void TestInputBow::setUp()
{
}

void TestInputBow::tearDown()
{
    int i;
    Component *comp = new DummyComponent1D("component", 10);
    InputBow *input = new InputBow(comp, 0.0, 0.0, 0.0,
				   0.0, 2.0, 5.0, 2.0, 1.0,
				   0.2);

    // ensure that it's set the first input timestep to 0
    CPPUNIT_ASSERT_EQUAL(0, Input::getFirstInputTimestep());
    comp->addInput(input);

    input->runTimestep(0, comp->getU(), comp->getU1(), comp->getU2());
    comp->swapBuffers(0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, comp->getU1()[0], 1e-12);

    for (i = 1; i < 100; i++) {
	input->runTimestep(i, comp->getU(), comp->getU1(), comp->getU2());
	comp->swapBuffers(i);
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.25137519193053e-10, comp->getU1()[0], 1e-20);

    // this will delete input along with it
    delete comp;

#ifdef USE_GPU
    double *d_u, *d_u1, *d_u2, *tmp;
    double val = 0.0;

    d_u = (double *)cudaMallocNess(sizeof(double));
    d_u1 = (double *)cudaMallocNess(sizeof(double));
    d_u2 = (double *)cudaMallocNess(sizeof(double));
    
    cudaMemcpyH2D(d_u, &val, sizeof(double));
    cudaMemcpyH2D(d_u1, &val, sizeof(double));
    cudaMemcpyH2D(d_u2, &val, sizeof(double));

    comp = new DummyComponent1D("component", 10);
    input = new InputBow(comp, 0.0, 0.0, 0.0,
			 0.0, 2.0, 5.0, 2.0, 1.0,
			 0.2);
    input->moveToGPU();

    input->runTimestep(0, d_u, d_u1, d_u2);
    tmp = d_u2;
    d_u2 = d_u1;
    d_u1 = d_u;
    d_u = tmp;
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, val, 1e-12);

    for (i = 1; i < 100; i++) {
	input->runTimestep(i, d_u, d_u1, d_u2);
	tmp = d_u2;
	d_u2 = d_u1;
	d_u1 = d_u;
	d_u = tmp;
    }
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.25137519193053e-10, val, 1e-20);

    cudaFreeNess(d_u);
    cudaFreeNess(d_u1);
    cudaFreeNess(d_u2);

    delete comp;
#endif
}

void TestInputBow::testInputBow()
{
}

