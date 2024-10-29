/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestMatlabFunction_StringDefGen.h"

void TestMatlabFunction_StringDefGen::setUp()
{
}

void TestMatlabFunction_StringDefGen::tearDown()
{
}

void TestMatlabFunction_StringDefGen::testMatlabFunction_StringDefGen()
{
    int i;
    MatlabCellContent result;
    MatlabCellContent material_tab, notes, materials, inharmonicity, L, T60_0, T60_1000;

    material_tab.type = CELL_ARRAY;
    material_tab.array.width = 2;
    material_tab.array.height = 1;
    material_tab.array.data = new double[2];
    material_tab.array.data[0] = 7850.0;
    material_tab.array.data[1] = 2e11;

    notes.type = CELL_ARRAY;
    notes.array.width = 6;
    notes.array.height = 1;
    notes.array.data = new double[6];
    notes.array.data[0] = -8;
    notes.array.data[1] = -3;
    notes.array.data[2] = 2;
    notes.array.data[3] = 7;
    notes.array.data[4] = 11;
    notes.array.data[5] = 16;

    materials.type = CELL_ARRAY;
    materials.array.width = 6;
    materials.array.height = 1;
    materials.array.data = new double[6];

    inharmonicity.type = CELL_ARRAY;
    inharmonicity.array.width = 6;
    inharmonicity.array.height = 1;
    inharmonicity.array.data = new double[6];
    inharmonicity.array.data[0] = 0.00001;
    inharmonicity.array.data[1] = 0.00001;
    inharmonicity.array.data[2] = 0.000001;
    inharmonicity.array.data[3] = 0.000001;
    inharmonicity.array.data[4] = 0.00001;
    inharmonicity.array.data[5] = 0.00001;

    L.type = CELL_SCALAR;
    L.scalar.value = 0.68;

    T60_0.type = CELL_ARRAY;
    T60_0.array.width = 6;
    T60_0.array.height = 1;
    T60_0.array.data = new double[6];
    
    T60_1000.type = CELL_ARRAY;
    T60_1000.array.width = 6;
    T60_1000.array.height = 1;
    T60_1000.array.data = new double[6];

    for (i = 0; i < 6; i++) {
	materials.array.data[i] = 1.0;
	T60_0.array.data[i] = 15.0;
	T60_1000.array.data[i] = 7.0;
    }

    MatlabFunction_StringDefGen *func = new MatlabFunction_StringDefGen();
    func->setParameter(0, material_tab);
    func->setParameter(1, notes);
    func->setParameter(2, materials);
    func->setParameter(3, inharmonicity);
    func->setParameter(4, L);
    func->setParameter(5, T60_0);
    func->setParameter(6, T60_1000);

    CPPUNIT_ASSERT(func->execute(&result));

    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(7, result.array.width);
    CPPUNIT_ASSERT_EQUAL(6, result.array.height);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.68, result.array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2e11, result.array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.577171644810595, result.array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000060785340030, result.array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7850.0, result.array.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, result.array.data[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, result.array.data[6], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.68, result.array.data[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2e11, result.array.data[8], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(14.531614168022703, result.array.data[9], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000081138694422, result.array.data[10], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7850.0, result.array.data[11], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, result.array.data[12], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, result.array.data[13], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.68, result.array.data[14], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2e11, result.array.data[15], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.613499923423044, result.array.data[16], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000034249732208, result.array.data[17], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7850.0, result.array.data[18], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, result.array.data[19], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, result.array.data[20], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.68, result.array.data[21], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2e11, result.array.data[22], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(14.646949263393905, result.array.data[23], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000045717907546, result.array.data[24], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7850.0, result.array.data[25], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, result.array.data[26], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, result.array.data[27], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.68, result.array.data[28], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2e11, result.array.data[29], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(369.079993873843820, result.array.data[30], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000182150210276, result.array.data[31], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7850.0, result.array.data[32], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, result.array.data[33], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, result.array.data[34], 1e-10);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.68, result.array.data[35], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2e11, result.array.data[36], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1171.755941071511600, result.array.data[37], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000243141360121, result.array.data[38], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7850.0, result.array.data[39], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(15.0, result.array.data[40], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, result.array.data[41], 1e-10);

    delete[] material_tab.array.data;
    delete[] notes.array.data;
    delete[] materials.array.data;
    delete[] inharmonicity.array.data;
    delete[] T60_0.array.data;
    delete[] T60_1000.array.data;
    delete[] result.array.data;
    delete func;
}

