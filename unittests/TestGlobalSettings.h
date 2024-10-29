/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Unit tests for GlobalSettings
 */
#ifndef _TESTGLOBALSETTINGS_H_
#define _TESTGLOBALSETTINGS_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "GlobalSettings.h"

class TestGlobalSettings : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testGlobalSettings();

    static CppUnit::Test *suite() {
	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("GlobalSettingsTest");
	suiteOfTests->addTest(new CppUnit::TestCaller<TestGlobalSettings>("testGlobalSettings",
									  &TestGlobalSettings::testGlobalSettings));
	return suiteOfTests;
    }

};

#endif
