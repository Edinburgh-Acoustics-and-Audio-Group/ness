/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for GPUUtil
 */
#ifndef _TESTGPUUTIL_H_
#define _TESTGPUUTIL_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "GPUUtil.h"

class TestGPUUtil : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testGPUUtil();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("GPUUtilTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestGPUUtil>("testGPUUtil", &TestGPUUtil::testGPUUtil));
        return suiteOfTests;
    }

};

#endif
