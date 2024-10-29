/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestBreakpointFunction.h"

void TestBreakpointFunction::setUp()
{
}

void TestBreakpointFunction::tearDown()
{
}

void TestBreakpointFunction::testBreakpointFunction()
{
    // test basic version
    static double x[4] = { 2.0, 5.0, 6.5, 8.0 };
    static double v[4] = { 3.0, -1.5, 2.6, 10.0 };

    BreakpointFunction *bpf = new BreakpointFunction(x, v, 4, 1.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.5, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2333333333333333333333333, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0666666666666666666666667, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, bpf->getValue(), 1e-12);
    bpf->next();
    delete bpf;

    // test interleaved version, and different value for k
    static double xandv[8] = { 4.0, 3.0, 10.0, -1.5, 13.0, 2.6, 16.0, 10.0 };

    bpf = new BreakpointFunction(xandv, 4, 2.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.5, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.2333333333333333333333333, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0666666666666666666666667, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, bpf->getValue(), 1e-12);
    bpf->next();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, bpf->getValue(), 1e-12);
    bpf->next();
    delete bpf;
}

