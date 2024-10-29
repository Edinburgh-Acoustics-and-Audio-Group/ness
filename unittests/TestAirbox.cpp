/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestAirbox.h"
#include "DummyAirbox.h"

void TestAirbox::setUp()
{
    airbox = new DummyAirbox("airbox", 1.2, 1.2, 1.2, 340.0, 1.21, 0.0);
}

void TestAirbox::tearDown()
{
    delete airbox;
}

void TestAirbox::testAirbox()
{
    // check the values that we passed in
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2, airbox->getLX(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2, airbox->getLY(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2, airbox->getLZ(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.21, airbox->getRhoA(), 1.0e-12);
    
    // check computed values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0134831460674157, airbox->getQ(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.571806500377929, airbox->getGamma(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, airbox->getLambda(), 1.0e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.26757369614512e-5, airbox->getK(), 1.0e-12);

    // check state array size
    CPPUNIT_ASSERT_EQUAL(90, airbox->getNx());
    CPPUNIT_ASSERT_EQUAL(90, airbox->getNy());
    CPPUNIT_ASSERT_EQUAL(90, airbox->getNz());
}

