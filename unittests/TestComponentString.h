/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for ComponentString
 */
#ifndef _TESTCOMPONENTSTRING_H_
#define _TESTCOMPONENTSTRING_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "ComponentString.h"

class TestComponentString : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testComponentString();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("ComponentStringTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestComponentString>("testComponentString", &TestComponentString::testComponentString));
        return suiteOfTests;
    }

 private:
    ComponentString *str;
};

#endif
