/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for SettingsManager
 */
#ifndef _TESTSETTINGSMANAGER_H_
#define _TESTSETTINGSMANAGER_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "SettingsManager.h"

class TestSettingsManager : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testSettingsManager();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("SettingsManagerTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestSettingsManager>("testSettingsManager", &TestSettingsManager::testSettingsManager));
        return suiteOfTests;
    }

};

#endif
