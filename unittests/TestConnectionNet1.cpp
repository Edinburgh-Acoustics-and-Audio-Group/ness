/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestConnectionNet1.h"
#include "DummyComponent1D.h"
#include "GlobalSettings.h"

void TestConnectionNet1::setUp()
{
}

void TestConnectionNet1::tearDown()
{
}

void TestConnectionNet1::testConnectionNet1()
{
    GlobalSettings::getInstance()->setSampleRate(48000);

    int i;
    DummyComponent1D *comp1 = new DummyComponent1D("comp1", 10);

    // test single ended connection
    ConnectionNet1 *conn = new ConnectionNet1(comp1, NULL, 0.005, 7000.0, 12.0,
					      2.4, 0.000003, 0.0, 0.0);
    comp1->getU()[0] = 0.001;
    comp1->getU1()[0] = 0.001;
    comp1->getU2()[0] = 0.001;

    for (i = 0; i < 10; i++) {
	conn->runTimestep(i);
	comp1->swapBuffers(i);
    }
    
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000999997193898412, comp1->getU1()[0], 1e-10);

    delete conn;

    // test double ended connection
    DummyComponent1D *comp2 = new DummyComponent1D("comp2", 10);
    conn = new ConnectionNet1(comp1, comp2, 0.005, 7000.0, 12.0,
			      2.4, 0.000003, 0.0, 0.0);

    comp1->getU()[0] = 0.001;
    comp1->getU1()[0] = 0.001;
    comp1->getU2()[0] = 0.001;

    comp2->getU()[0] = 0.001;
    comp2->getU1()[0] = 0.001;
    comp2->getU2()[0] = 0.001;

    for (i = 0; i < 10; i++) {
	conn->runTimestep(i);
	comp1->swapBuffers(i);
	comp2->swapBuffers(i);
    }
    
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000999994834249755, comp1->getU1()[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.000999994834249755, comp2->getU1()[0], 1e-10);

    delete conn;
    delete comp1;
    delete comp2;
}
