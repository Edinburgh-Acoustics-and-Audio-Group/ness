/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestMatlabFunction_PluckGen.h"

void TestMatlabFunction_PluckGen::setUp()
{
}

void TestMatlabFunction_PluckGen::tearDown()
{
}

void TestMatlabFunction_PluckGen::testMatlabFunction_PluckGen()
{
    MatlabFunction_PluckGen *func = new MatlabFunction_PluckGen();
    MatlabCellContent exc, param, result;

    exc.type = CELL_ARRAY;
    exc.array.width = 5;
    exc.array.height = 1;
    exc.array.data = new double[5];
    exc.array.data[0] = 2.0;
    exc.array.data[1] = 0.6;
    exc.array.data[2] = 0.5;
    exc.array.data[3] = 0.003;
    exc.array.data[4] = 1.5;
    func->setParameter(0, exc);

    param.type = CELL_SCALAR;
    param.scalar.value = 1.0;
    func->setParameter(1, param);

    param.scalar.value = 0.9;
    func->setParameter(2, param);

    param.scalar.value = 0.8;
    func->setParameter(3, param);

    param.scalar.value = 0.001;
    func->setParameter(4, param);

    param.scalar.value = 1.0;
    func->setParameter(5, param);

    CPPUNIT_ASSERT(func->execute(&result));
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(5, result.array.width);
    CPPUNIT_ASSERT_EQUAL(2, result.array.height);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6, result.array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, result.array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.003, result.array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, result.array.data[4], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.array.data[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9, result.array.data[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, result.array.data[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.001, result.array.data[8], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.array.data[9], 1e-10);

    delete[] exc.array.data;
    delete[] result.array.data;
    delete func;
}

