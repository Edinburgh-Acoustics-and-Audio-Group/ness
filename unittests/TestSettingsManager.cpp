/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestSettingsManager.h"

#include "SettingsManager.h"

void TestSettingsManager::setUp()
{
}

void TestSettingsManager::tearDown()
{
}

void TestSettingsManager::testSettingsManager()
{
    SettingsManager *sm = SettingsManager::getInstance();

    // try getting fixpar setting, it should be default
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, sm->getDoubleSetting("smtestcomp1", "fixpar"), 1e-12);

    // now add a setting for it for that component
    sm->putSetting("smtestcomp1", "fixpar", 1.4);

    // check it worked
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.4, sm->getDoubleSetting("smtestcomp1", "fixpar"), 1e-12);

    // check it still falls back to default for a different component
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, sm->getDoubleSetting("smtestcomp2", "fixpar"), 1e-12);

    // test the argument parsing version of putSetting
    sm->putSetting("-testsetting:smtestcomp3", "42");
    CPPUNIT_ASSERT_EQUAL(42, sm->getIntSetting("smtestcomp3", "testsetting"));
    CPPUNIT_ASSERT("42" == sm->getStringSetting("smtestcomp3", "testsetting"));
    CPPUNIT_ASSERT_EQUAL(0, sm->getIntSetting("smtestcomp4", "testsetting"));

    // try for boolean
    sm->putSetting("-testsetting2:smtestcomp3", "true");
    CPPUNIT_ASSERT(sm->getBoolSetting("smtestcomp3", "testsetting2"));
}
