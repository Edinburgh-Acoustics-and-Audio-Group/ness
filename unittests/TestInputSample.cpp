/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestInputSample.h"

#include "DummyComponent1D.h"
#include "DummyComponent2D.h"
#include "DummyComponent3D.h"
#include "DummyInputSample.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

void TestInputSample::setUp()
{
}

void TestInputSample::tearDown()
{
}

void TestInputSample::testInputSample()
{
    int i;
    double *data = new double[10];
    for (i = 0; i < 10; i++) data[i] = (double)i;

    // test basic input
    DummyComponent1D *comp1d = new DummyComponent1D("comp1d", 10);
    DummyInputSample *input = new DummyInputSample(comp1d, 2.0, 0.05, 0.0, 0.0, data, 10, 0, 0);

    comp1d->addInput(input);

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, comp1d->getU(), comp1d->getU1(),
			   comp1d->getU2());
	comp1d->swapBuffers(i);
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, comp1d->getU1()[0], 1e-10);

    delete comp1d;

    // test negated
    comp1d = new DummyComponent1D("comp1d", 10);
    input = new DummyInputSample(comp1d, 2.0, 0.05, 0.0, 0.0, data, 10, 0, 1);

    comp1d->addInput(input);

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, comp1d->getU(), comp1d->getU1(),
			   comp1d->getU2());
	comp1d->swapBuffers(i);
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-10.0, comp1d->getU1()[0], 1e-10);

    delete comp1d;

    // test linear interp
    comp1d = new DummyComponent1D("comp1d", 10);
    input = new DummyInputSample(comp1d, 2.0, 0.05, 0.0, 0.0, data, 10, 1, 0);

    comp1d->addInput(input);

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, comp1d->getU(), comp1d->getU1(),
			   comp1d->getU2());
	comp1d->swapBuffers(i);
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, comp1d->getU1()[0], 1e-10);

    delete comp1d;

    // test bilinear interp
    DummyComponent2D *comp2d = new DummyComponent2D("comp2d", 10, 10);
    input = new DummyInputSample(comp2d, 2.0, 0.05, 0.05, 0.0, data, 10, 1, 0);

    comp2d->addInput(input);

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, comp2d->getU(), comp2d->getU1(),
			   comp2d->getU2());
	comp2d->swapBuffers(i);
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, comp2d->getU1()[0], 1e-10);

    delete comp2d;

    // test trilinear interp
    DummyComponent3D *comp3d = new DummyComponent3D("comp3d", 10, 10, 10);
    input = new DummyInputSample(comp3d, 2.0, 0.05, 0.05, 0.05, data, 10, 1, 0);

    comp3d->addInput(input);

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, comp3d->getU(), comp3d->getU1(),
			   comp3d->getU2());
	comp3d->swapBuffers(i);
    }
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.25, comp3d->getU1()[0], 1e-10);

    delete comp3d;

    // test GPU version
#ifdef USE_GPU
    double *d_u, *d_u1, *d_u2, *tmp;
    double val = 0.0;

    d_u = (double *)cudaMallocNess(1000 * sizeof(double));
    d_u1 = (double *)cudaMallocNess(1000 * sizeof(double));
    d_u2 = (double *)cudaMallocNess(1000 * sizeof(double));
    
    cudaMemsetNess(d_u, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u1, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u2, 0, 1000 * sizeof(double));

    // test basic input
    comp1d = new DummyComponent1D("comp1d", 10);
    input = new DummyInputSample(comp1d, 2.0, 0.05, 0.0, 0.0, data, 10, 0, 0);

    comp1d->addInput(input);
    input->moveToGPU();

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, d_u, d_u1, d_u2);
	tmp = d_u2;
	d_u2 = d_u1;
	d_u1 = d_u;
	d_u = tmp;
    }
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, val, 1e-10);

    delete comp1d;

    // test negated
    comp1d = new DummyComponent1D("comp1d", 10);
    input = new DummyInputSample(comp1d, 2.0, 0.05, 0.0, 0.0, data, 10, 0, 1);

    comp1d->addInput(input);
    input->moveToGPU();

    cudaMemsetNess(d_u, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u1, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u2, 0, 1000 * sizeof(double));

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, d_u, d_u1, d_u2);
	tmp = d_u2;
	d_u2 = d_u1;
	d_u1 = d_u;
	d_u = tmp;
    }
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-10.0, val, 1e-10);

    delete comp1d;

    // test linear interpolated
    comp1d = new DummyComponent1D("comp1d", 10);
    input = new DummyInputSample(comp1d, 2.0, 0.05, 0.0, 0.0, data, 10, 1, 0);

    comp1d->addInput(input);
    input->moveToGPU();

    cudaMemsetNess(d_u, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u1, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u2, 0, 1000 * sizeof(double));

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, d_u, d_u1, d_u2);
	tmp = d_u2;
	d_u2 = d_u1;
	d_u1 = d_u;
	d_u = tmp;
    }
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, val, 1e-10);

    delete comp1d;

    // test bilinear interpolated
    comp2d = new DummyComponent2D("comp2d", 10, 10);
    input = new DummyInputSample(comp2d, 2.0, 0.05, 0.05, 0.0, data, 10, 1, 0);

    comp2d->addInput(input);
    input->moveToGPU();

    cudaMemsetNess(d_u, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u1, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u2, 0, 1000 * sizeof(double));

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, d_u, d_u1, d_u2);
	tmp = d_u2;
	d_u2 = d_u1;
	d_u1 = d_u;
	d_u = tmp;
    }
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, val, 1e-10);

    delete comp2d;

    // test trilinear interpolated
    comp3d = new DummyComponent3D("comp3d", 10, 10, 10);
    input = new DummyInputSample(comp3d, 2.0, 0.05, 0.05, 0.05, data, 10, 1, 0);

    comp3d->addInput(input);
    input->moveToGPU();

    cudaMemsetNess(d_u, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u1, 0, 1000 * sizeof(double));
    cudaMemsetNess(d_u2, 0, 1000 * sizeof(double));

    for (i = 0; i < 5; i++) {
	input->runTimestep(i, d_u, d_u1, d_u2);
	tmp = d_u2;
	d_u2 = d_u1;
	d_u1 = d_u;
	d_u = tmp;
    }
    cudaMemcpyD2H(&val, d_u1, sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.25, val, 1e-10);

    delete comp3d;

    cudaFreeNess(d_u);
    cudaFreeNess(d_u1);
    cudaFreeNess(d_u2);
#endif

    delete[] data;
}

