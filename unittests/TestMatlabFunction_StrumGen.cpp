/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestMatlabFunction_StrumGen.h"
#include "InstrumentParserGuitar.h"

#include <cstdlib>
using namespace std;

void TestMatlabFunction_StrumGen::setUp()
{
}

void TestMatlabFunction_StrumGen::tearDown()
{
}

static const double expectedResult[] = {
    2.000000000000000, 0.600000000000000, 0.500000000000000, 0.003000000000000, 1.500000000000000,
    6.000000000000000, 0.000000000000000, 0.791550634145528, 0.001028309922376, 1.029844003347607,
    5.000000000000000, 0.083293178863494, 0.775804109543471, 0.000983522275571, 1.026822959481190,
    4.000000000000000, 0.156444395372851, 0.804317596463634, 0.000997739705186, 1.012887092476192,
    3.000000000000000, 0.236754827347004, 0.801072072815649, 0.001045222972517, 1.041619506800370,
    2.000000000000000, 0.324342775294717, 0.817383754354615, 0.000964160255536, 1.010696887625706,
    1.000000000000000, 0.380652022864973, 0.779430941650379, 0.000963723157679, 1.030417675422699
};

void TestMatlabFunction_StrumGen::testMatlabFunction_StrumGen()
{
    // reset RNG to something predictable
    srand(0);

    InstrumentParserGuitar::setNumStrings(6);

    MatlabFunction_StrumGen *func = new MatlabFunction_StrumGen();
    MatlabCellContent exc, param, result;
    int i;

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
    param.scalar.value = 0.0;
    func->setParameter(1, param);

    param.scalar.value = 0.4;
    func->setParameter(2, param);

    param.scalar.value = 1.0;
    func->setParameter(3, param);

    param.scalar.value = 1.0;
    func->setParameter(4, param);

    param.scalar.value = 0.1;
    func->setParameter(5, param);

    param.scalar.value = 0.001;
    func->setParameter(6, param);

    param.scalar.value = 0.1;
    func->setParameter(7, param);

    param.scalar.value = 0.8;
    func->setParameter(8, param);

    param.scalar.value = 0.1;
    func->setParameter(9, param);

    param.scalar.value = 0.1;
    func->setParameter(10, param);

    CPPUNIT_ASSERT(func->execute(&result));

    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(5, result.array.width);
    CPPUNIT_ASSERT_EQUAL(7, result.array.height);
    
    for (i = 0; i < (7*5); i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedResult[i], result.array.data[i], 1e-10);
    }

    delete[] exc.array.data;
    delete[] result.array.data;
    delete func;
}

