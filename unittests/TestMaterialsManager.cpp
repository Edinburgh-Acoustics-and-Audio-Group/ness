/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */

#include "TestMaterialsManager.h"

void TestMaterialsManager::setUp()
{
    mm = MaterialsManager::getInstance();
    mm->loadMaterials("materials.txt");
}

void TestMaterialsManager::tearDown()
{
}

void TestMaterialsManager::testMaterialsManager()
{
    CPPUNIT_ASSERT(mm != NULL);
    CPPUNIT_ASSERT(mm->getMaterial("nonexist") == NULL);

    Material *lead = mm->getMaterial("lead");
    CPPUNIT_ASSERT(lead != NULL);
    CPPUNIT_ASSERT(lead->getName() == "lead");
    CPPUNIT_ASSERT_EQUAL(1.6e10, lead->getYoungsModulus());
    CPPUNIT_ASSERT_EQUAL(11340.0, lead->getDensity());

    Material *steel = mm->getMaterial("steel");
    CPPUNIT_ASSERT(steel != NULL);
    CPPUNIT_ASSERT_EQUAL(0.3, steel->getPoissonsRatio());
}
