/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestMatlabFunction_FretDefGen.h"

void TestMatlabFunction_FretDefGen::setUp()
{
}

void TestMatlabFunction_FretDefGen::tearDown()
{
}

void TestMatlabFunction_FretDefGen::testMatlabFunction_FretDefGen()
{
    // test 3-parameter version
    MatlabFunction_FretDefGen *func = new MatlabFunction_FretDefGen();
    MatlabCellContent param, result;
    param.type = CELL_SCALAR;

    param.scalar.value = 4.0;
    func->setParameter(0, param);

    param.scalar.value = 1.0;
    func->setParameter(1, param);

    param.scalar.value = -0.001;
    func->setParameter(2, param);

    CPPUNIT_ASSERT(func->execute(&result));
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(2, result.array.width);
    CPPUNIT_ASSERT_EQUAL(4, result.array.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0561256873183065, result.array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.109101281859661, result.array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.159103584746285, result.array.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2062994740159, result.array.data[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[7], 1e-10);
    delete func;
    delete[] result.array.data;

    // test 4-parameter version
    func = new MatlabFunction_FretDefGen();

    param.type = CELL_ARRAY;
    param.array.width = 2;
    param.array.height = 1;
    param.array.data = new double[2];
    param.array.data[0] = 0.2;
    param.array.data[1] = -0.0015;
    func->setParameter(0, param);

    param.type = CELL_SCALAR;
    param.scalar.value = 4.0;
    func->setParameter(1, param);

    param.scalar.value = 1.0;
    func->setParameter(2, param);

    param.scalar.value = -0.001;
    func->setParameter(3, param);

    CPPUNIT_ASSERT(func->execute(&result));
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(2, result.array.width);
    CPPUNIT_ASSERT_EQUAL(5, result.array.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, result.array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0015, result.array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0561256873183065, result.array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.109101281859661, result.array.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.159103584746285, result.array.data[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2062994740159, result.array.data[8], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, result.array.data[9], 1e-10);
    delete func;
    delete[] result.array.data;
    delete[] param.array.data;
}

