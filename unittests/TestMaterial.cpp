/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */

#include "TestMaterial.h"

void TestMaterial::setUp()
{
    mat = new Material("testmaterial", 1000.0, 13.3, 200.9);
}

void TestMaterial::tearDown()
{
    delete mat;
}

void TestMaterial::testMaterial()
{
    CPPUNIT_ASSERT(mat->getName() == "testmaterial");
    CPPUNIT_ASSERT_EQUAL(1000.0, mat->getYoungsModulus());
    CPPUNIT_ASSERT_EQUAL(13.3, mat->getPoissonsRatio());
    CPPUNIT_ASSERT_EQUAL(200.9, mat->getDensity());
}

