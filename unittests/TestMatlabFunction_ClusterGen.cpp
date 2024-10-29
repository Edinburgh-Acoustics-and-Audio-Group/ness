/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestMatlabFunction_ClusterGen.h"
#include "InstrumentParserGuitar.h"

#include <cstdlib>
using namespace std;

void TestMatlabFunction_ClusterGen::setUp()
{
}

void TestMatlabFunction_ClusterGen::tearDown()
{
}

static const double expectedResult[] = {
    2.000000000000000, 0.600000000000000, 0.500000000000000, 0.003000000000000, 1.500000000000000,
    1.000000000000000, 0.768037543430942, 0.791550634145528, 0.002056619844752, 1.029844003347607,
    1.000000000000000, 0.782329471587357, 0.775804109543471, 0.001967044551143, 1.026822959481190,
    1.000000000000000, 0.655554942160637, 0.804317596463634, 0.001995479410372, 1.012887092476192,
    1.000000000000000, 0.672956894558369, 0.801072072815649, 0.002090445945035, 1.041619506800370,
    1.000000000000000, 0.727142345591980, 0.817383754354615, 0.001928320511071, 1.010696887625706,
    1.000000000000000, 0.603260114324866, 0.779430941650379, 0.001927446315357, 1.030417675422699,
    1.000000000000000, 0.631335817850817, 0.792075551539695, 0.001925958089356, 0.960880880202577,
    1.000000000000000, 0.799784903600712, 0.777460552424873, 0.002002586478881, 1.033911223469261,
    1.000000000000000, 0.722527966519132, 0.783682529415787, 0.002027510453541, 1.002428719006679,
    1.000000000000000, 0.698716597398145, 0.837822001910686, 0.001958503356883, 1.027135769779391,
    1.000000000000000, 0.705348995842668, 0.821593106902015, 0.001980045724418, 1.039152945200518,
    1.000000000000000, 0.656662949201028, 0.788196667781191, 0.002061544904002, 1.041902647396504,
    1.000000000000000, 0.613951055246382, 0.835946166029175, 0.002005199070044, 0.958605584785624,
    1.000000000000000, 0.638442769198885, 0.813058154160650, 0.002078046520510, 0.984889293524851,
    1.000000000000000, 0.612834264157728, 0.761601843909175, 0.001991540347455, 0.956309583832654,
    1.000000000000000, 0.647655990835119, 0.837650730534294, 0.002080441614697, 1.035091978677126,
    1.000000000000000, 0.653333149875204, 0.803180827257773, 0.001975041395274, 1.026024873636675,
    1.000000000000000, 0.702507072828015, 0.813417900862832, 0.002006321286832, 0.953928034335341,
    1.000000000000000, 0.687527519318986, 0.834546804500067, 0.002086161959072, 1.022095234306573,
    1.000000000000000, 0.656858680610014, 0.819082745192145, 0.002027995763313, 0.985404867974764
};

void TestMatlabFunction_ClusterGen::testMatlabFunction_ClusterGen()
{
    // reset RNG to something predictable
    srand(0);

    InstrumentParserGuitar::setNumStrings(0);

    MatlabFunction_ClusterGen *func = new MatlabFunction_ClusterGen();
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
    param.scalar.value = 0.6;
    func->setParameter(1, param);

    param.scalar.value = 20.0;
    func->setParameter(2, param);

    param.scalar.value = 0.2;
    func->setParameter(3, param);

    param.scalar.value = 1.0;
    func->setParameter(4, param);

    param.scalar.value = 0.1;
    func->setParameter(5, param);

    param.scalar.value = 0.002;
    func->setParameter(6, param);

    param.scalar.value = 0.1;
    func->setParameter(7, param);

    param.scalar.value = 0.8;
    func->setParameter(8, param);

    param.scalar.value = 0.1;
    func->setParameter(9, param);

    CPPUNIT_ASSERT(func->execute(&result));
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(5, result.array.width);
    CPPUNIT_ASSERT_EQUAL(21, result.array.height);
    
    for (i = 0; i < (21*5); i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedResult[i], result.array.data[i], 1e-10);
    }

    delete[] exc.array.data;
    delete[] result.array.data;
    delete func;
}

