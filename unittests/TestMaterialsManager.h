/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Unit tests for MaterialsManager
 */

#ifndef _TESTMATERIALSMANAGER_H_
#define _TESTMATERIALSMANAGER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "MaterialsManager.h"

class TestMaterialsManager : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMaterialsManager();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MaterialsManagerTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestMaterialsManager>("testMaterialsManager",
									    &TestMaterialsManager::testMaterialsManager));
	return suiteOfTests;
    }

    private:
	MaterialsManager *mm;
};

#endif
