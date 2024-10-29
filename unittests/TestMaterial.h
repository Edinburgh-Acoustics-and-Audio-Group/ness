/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Unit tests for Material class
 */

#ifndef _TESTMATERIAL_H_
#define _TESTMATERIAL_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Material.h"

class TestMaterial : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testMaterial();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MaterialTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestMaterial>("testMaterial",
								    &TestMaterial::testMaterial));
	return suiteOfTests;
    }

 private:
    Material *mat;
};

#endif
