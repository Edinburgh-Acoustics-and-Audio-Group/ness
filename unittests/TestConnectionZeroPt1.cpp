/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestConnectionZeroPt1.h"

#include "DummyComponent1D.h"
#include "ConnectionZeroPt1.h"

#ifdef USE_GPU
#include "GPUUtil.h"
#endif

void TestConnectionZeroPt1::setUp()
{
}

void TestConnectionZeroPt1::tearDown()
{
}

void TestConnectionZeroPt1::testConnectionZeroPt1()
{
    int i;
    DummyComponent1D *comp1 = new DummyComponent1D("comp1", 10);
    DummyComponent1D *comp2 = new DummyComponent1D("comp2", 10);
    ConnectionZeroPt1 *conn = new ConnectionZeroPt1(comp1, comp2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
						    100000.0, 1.0, 1.0, 0.694828622975817);

    comp1->getU1()[0] = 0.4;
    comp2->getU1()[0] = 0.6;

    for (i = 0; i < 10; i++) {
	conn->runTimestep(i);
	comp1->swapBuffers(i);
	comp2->swapBuffers(i);
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-11.1070535286673, comp1->getU1()[0], 1.0e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11.1070535286673, comp2->getU1()[0], 1.0e-10);    

    delete comp1;
    delete comp2;
    delete conn;


#ifdef USE_GPU
    // now test full GPU version
    comp1 = new DummyComponent1D("comp1", 10);
    comp2 = new DummyComponent1D("comp2", 10);
    conn = new ConnectionZeroPt1(comp1, comp2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				 100000.0, 1.0, 1.0, 0.694828622975817);

    comp1->moveToGPU();
    comp2->moveToGPU();
    conn->maybeMoveToGPU();

    double val = 0.4;
    cudaMemcpyH2D(comp1->getU1(), &val, sizeof(double));
    val = 0.6;
    cudaMemcpyH2D(comp2->getU1(), &val, sizeof(double));

    for (i = 0; i < 10; i++) {
	conn->runTimestep(i);
	comp1->swapBuffers(i);
	comp2->swapBuffers(i);
    }

    cudaMemcpyD2H(&val, comp1->getU1(), sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-11.1070535286673, val, 1.0e-10);
    cudaMemcpyD2H(&val, comp2->getU1(), sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11.1070535286673, val, 1.0e-10);    
    
    delete comp1;
    delete comp2;
    delete conn;

    // test with one component on CPU, one on GPU
    comp1 = new DummyComponent1D("comp1", 10);
    comp2 = new DummyComponent1D("comp2", 10);
    conn = new ConnectionZeroPt1(comp1, comp2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				 100000.0, 1.0, 1.0, 0.694828622975817);

    comp1->moveToGPU();
    conn->maybeMoveToGPU();

    val = 0.4;
    cudaMemcpyH2D(comp1->getU1(), &val, sizeof(double));
    comp2->getU1()[0] = 0.6;

    for (i = 0; i < 10; i++) {
	conn->runTimestep(i);
	comp1->swapBuffers(i);
	comp2->swapBuffers(i);
    }

    cudaMemcpyD2H(&val, comp1->getU1(), sizeof(double));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-11.1070535286673, val, 1.0e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11.1070535286673, comp2->getU1()[0], 1.0e-10);    
    
    delete comp1;
    delete comp2;
    delete conn;
#endif
}

