/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestOutputPressure.h"

#include "DummyComponent1D.h"
#include "DummyComponent2D.h"
#include "DummyComponent3D.h"
#include "OutputPressure.h"
#include "GlobalSettings.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

#include <cstring>
using namespace std;

void TestOutputPressure::setUp()
{
}

void TestOutputPressure::tearDown()
{
}

void TestOutputPressure::testOutputPressure()
{
    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    // test standard output
    DummyComponent1D *comp1d = new DummyComponent1D("comp1d", 10);
    Output *output = new OutputPressure(comp1d, 0.5, 0.05, 0.0, 0.0, 0);
    double *u = comp1d->getU();
    double *u1 = comp1d->getU2();
    double *u2 = comp1d->getU1();

    u[0] = 1.0;
    u1[0] = 0.5;
    u2[0] = 7.0;
    output->runTimestep(0);
    u[0] = 2.0;
    u1[0] = 1.0;
    output->runTimestep(1);
    u[0] = 3.0;
    output->runTimestep(2);
    u[0] = 4.0;
    output->runTimestep(3);

    double *data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(22050.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(88200.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(132300.0, data[3], 1e-10);

    // test max value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(132300.0, output->getMaxValue(),
				 1e-10);
    delete output;
    delete comp1d;


    // test linear interpolation
    comp1d = new DummyComponent1D("comp1d", 10);
    output = new OutputPressure(comp1d, 0.5, 0.05, 0.0, 0.0, 1);
    u = comp1d->getU();
    u1 = comp1d->getU2();

    u[0] = 1.0;
    u1[0] = 1.0;
    output->runTimestep(0);
    u[0] = 2.0;
    output->runTimestep(1);
    u[0] = 3.0;
    output->runTimestep(2);
    u[0] = 4.0;
    output->runTimestep(3);

    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(22050.0, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(66150.0, data[3], 1e-10);

    delete output;
    delete comp1d;

    
    // test bilinear interpolation
    DummyComponent2D *comp2d = new DummyComponent2D("comp2d", 10, 10);
    output = new OutputPressure(comp2d, 0.5, 0.05, 0.05, 0.0, 1);
    u = comp2d->getU();
    u1 = comp2d->getU2();

    u[0] = 1.0;
    u1[0] = 1.0;
    output->runTimestep(0);
    u[0] = 2.0;
    output->runTimestep(1);
    u[0] = 3.0;
    output->runTimestep(2);
    u[0] = 4.0;
    output->runTimestep(3);

    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11025.0, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(22050.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(33075.0, data[3], 1e-10);

    delete output;
    delete comp2d;

    
    // test trilinear interpolation
    DummyComponent3D *comp3d = new DummyComponent3D("comp3d", 10, 10, 10);
    output = new OutputPressure(comp3d, 0.5, 0.05, 0.05, 0.05, 1);
    u = comp3d->getU();
    u1 = comp3d->getU2();

    u[0] = 1.0;
    u1[0] = 1.0;
    output->runTimestep(0);
    u[0] = 2.0;
    output->runTimestep(1);
    u[0] = 3.0;
    output->runTimestep(2);
    u[0] = 4.0;
    output->runTimestep(3);

    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5512.5, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11025.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(16537.5, data[3], 1e-10);

    delete output;
    delete comp3d;    

#ifdef USE_GPU
    // GPU tests - basic output
    comp1d = new DummyComponent1D("comp1d", 10);
    output = new OutputPressure(comp1d, 0.5, 0.05, 0.0, 0.0, 0);

    comp1d->moveToGPU();
    output->maybeMoveToGPU();

    u = comp1d->getU();
    u1 = comp1d->getU2();
    u2 = comp1d->getU1();

    double val = 1.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    cudaMemcpyH2D(u1, &val, sizeof(double));
    val = 7.0;
    cudaMemcpyH2D(u2, &val, sizeof(double));
    output->runTimestep(0);
    val = 2.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(1);
    val = 3.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(2);
    val = 4.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(3);

    output->copyFromGPU();
    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(88200.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(132300.0, data[3], 1e-10);

    delete output;
    delete comp1d;

    // interpolated on GPU
    comp1d = new DummyComponent1D("comp1d", 10);
    output = new OutputPressure(comp1d, 0.5, 0.05, 0.0, 0.0, 1);

    comp1d->moveToGPU();
    output->maybeMoveToGPU();

    u = comp1d->getU();
    u1 = comp1d->getU2();

    val = 1.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    cudaMemcpyH2D(u1, &val, sizeof(double));
    output->runTimestep(0);
    val = 2.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(1);
    val = 3.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(2);
    val = 4.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(3);

    output->copyFromGPU();
    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(22050.0, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(66150.0, data[3], 1e-10);

    delete output;
    delete comp1d;

    // bilinear interpolated on GPU
    comp2d = new DummyComponent2D("comp2d", 10, 10);
    output = new OutputPressure(comp2d, 0.5, 0.05, 0.05, 0.0, 1);

    comp2d->moveToGPU();
    output->maybeMoveToGPU();

    u = comp2d->getU();
    u1 = comp2d->getU2();

    val = 1.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    cudaMemcpyH2D(u1, &val, sizeof(double));
    output->runTimestep(0);
    val = 2.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(1);
    val = 3.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(2);
    val = 4.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(3);

    output->copyFromGPU();
    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11025.0, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(22050.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(33075.0, data[3], 1e-10);

    delete output;
    delete comp2d;

    // trilinear interpolated on GPU
    comp3d = new DummyComponent3D("comp3d", 10, 10, 10);
    output = new OutputPressure(comp3d, 0.5, 0.05, 0.05, 0.05, 1);

    comp3d->moveToGPU();
    output->maybeMoveToGPU();

    u = comp3d->getU();
    u1 = comp3d->getU2();

    val = 1.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    cudaMemcpyH2D(u1, &val, sizeof(double));
    output->runTimestep(0);
    val = 2.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(1);
    val = 3.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(2);
    val = 4.0;
    cudaMemcpyH2D(u, &val, sizeof(double));
    output->runTimestep(3);

    output->copyFromGPU();
    data = output->getData();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5512.5, data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11025.0, data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(16537.5, data[3], 1e-10);

    delete output;
    delete comp3d;

#endif
}

