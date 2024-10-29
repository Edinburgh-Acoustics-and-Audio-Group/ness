/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestMatlabParser.h"

void TestMatlabParser::setUp()
{
}

void TestMatlabParser::tearDown()
{
}

void TestMatlabParser::testMatlabParser()
{
    MatlabParser *mp = new MatlabParser("matlab.m");

    // check we can open and parse the file
    CPPUNIT_ASSERT(mp->parse());

    // check scalars are as expected
    vector<MatlabScalar> *scalars = mp->getScalars();
    CPPUNIT_ASSERT_EQUAL(6, (int)scalars->size());
    // pi is defined automatically by the parser and should be first
    CPPUNIT_ASSERT(scalars->at(0).name == "pi");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.1415927, scalars->at(0).value, 1e-6);
    CPPUNIT_ASSERT(scalars->at(1).name == "x");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.32, scalars->at(1).value, 1e-10);
    CPPUNIT_ASSERT(scalars->at(2).name == "y");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-7e-10, scalars->at(2).value, 1e-10);
    CPPUNIT_ASSERT(scalars->at(3).name == "z");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2, scalars->at(3).value, 1e-10);
    CPPUNIT_ASSERT(scalars->at(4).name == "w");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(66.0, scalars->at(4).value, 1e-10);
    CPPUNIT_ASSERT(scalars->at(5).name == "v");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, scalars->at(5).value, 1e-10);

    // check arrays are as expected
    vector<MatlabArray> *arrays = mp->getArrays();
    CPPUNIT_ASSERT_EQUAL(4, (int)arrays->size());

    MatlabArray arr = arrays->at(0);
    CPPUNIT_ASSERT(arr.name == "arr1D");
    CPPUNIT_ASSERT_EQUAL(5, arr.width);
    CPPUNIT_ASSERT_EQUAL(1, arr.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, arr.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2, arr.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3, arr.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4, arr.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, arr.data[4], 1e-10);

    arr = arrays->at(1);
    CPPUNIT_ASSERT(arr.name == "arr2D");
    CPPUNIT_ASSERT_EQUAL(3, arr.width);
    CPPUNIT_ASSERT_EQUAL(3, arr.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, arr.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6, arr.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7, arr.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8, arr.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9, arr.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10, arr.data[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11, arr.data[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(12, arr.data[7], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(13, arr.data[8], 1e-10);

    arr = arrays->at(2);
    CPPUNIT_ASSERT(arr.name == "frets");
    CPPUNIT_ASSERT_EQUAL(2, arr.width);
    CPPUNIT_ASSERT_EQUAL(4, arr.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0561256873183065, arr.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, arr.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.109101281859661, arr.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, arr.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.159103584746285, arr.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, arr.data[5], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2062994740159, arr.data[6], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.001, arr.data[7], 1e-10);

    arr = arrays->at(3);
    CPPUNIT_ASSERT(arr.name == "arr3");
    CPPUNIT_ASSERT_EQUAL(3, arr.width);
    CPPUNIT_ASSERT_EQUAL(2, arr.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, arr.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, arr.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, arr.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, arr.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11.0, arr.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(13.0, arr.data[5], 1e-10);


    // check cell arrays are as expected
    vector<MatlabCellArray> *cellarrays = mp->getCellArrays();
    MatlabCellArray ca = cellarrays->at(0);
    CPPUNIT_ASSERT(ca.name == "cellarray");
    CPPUNIT_ASSERT_EQUAL(2, ca.width);
    CPPUNIT_ASSERT_EQUAL(2, ca.height);

    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, ca.data[0].type);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1, ca.data[0].scalar.value, 1e-10);
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, ca.data[1].type);
    CPPUNIT_ASSERT_EQUAL(3, ca.data[1].array.width);
    CPPUNIT_ASSERT_EQUAL(2, ca.data[1].array.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3, ca.data[1].array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, ca.data[1].array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7, ca.data[1].array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9, ca.data[1].array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11, ca.data[1].array.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(13, ca.data[1].array.data[5], 1e-10);

    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, ca.data[2].type);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2, ca.data[2].scalar.value, 1e-10);
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, ca.data[3].type);
    CPPUNIT_ASSERT_EQUAL(3, ca.data[3].array.width);
    CPPUNIT_ASSERT_EQUAL(2, ca.data[3].array.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4, ca.data[3].array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6, ca.data[3].array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8, ca.data[3].array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10, ca.data[3].array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(12, ca.data[3].array.data[4], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(14, ca.data[3].array.data[5], 1e-10);

    // check structures are as expected
    vector<MatlabStruct> *structs = mp->getStructs();
    CPPUNIT_ASSERT_EQUAL(2, (int)structs->size());
    MatlabStruct ms = structs->at(0);
    CPPUNIT_ASSERT(ms.name == "str1");
    CPPUNIT_ASSERT_EQUAL(1, (int)ms.elements.size());
    MatlabStructElement mse = ms.elements[0];
    CPPUNIT_ASSERT_EQUAL(4, (int)mse.memberNames.size());
    CPPUNIT_ASSERT_EQUAL(4, (int)mse.memberValues.size());
    CPPUNIT_ASSERT(mse.memberNames[0] == "a");
    CPPUNIT_ASSERT(mse.memberNames[1] == "b");
    CPPUNIT_ASSERT(mse.memberNames[2] == "c");
    CPPUNIT_ASSERT(mse.memberNames[3] == "d");
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[0].type);
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[1].type);
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[2].type);
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, mse.memberValues[3].type);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, mse.memberValues[0].scalar.value, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.32, mse.memberValues[1].scalar.value, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mse.memberValues[2].scalar.value, 1e-10);
    CPPUNIT_ASSERT_EQUAL(5, (int)mse.memberValues[3].array.width);
    CPPUNIT_ASSERT_EQUAL(1, (int)mse.memberValues[3].array.height);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(9.0, mse.memberValues[3].array.data[0], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mse.memberValues[3].array.data[1], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, mse.memberValues[3].array.data[2], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, mse.memberValues[3].array.data[3], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, mse.memberValues[3].array.data[4], 1e-10);

    ms = structs->at(1);
    CPPUNIT_ASSERT(ms.name == "str2");
    CPPUNIT_ASSERT_EQUAL(2, (int)ms.elements.size());
    mse = ms.elements[0];
    CPPUNIT_ASSERT_EQUAL(2, (int)mse.memberNames.size());
    CPPUNIT_ASSERT_EQUAL(2, (int)mse.memberValues.size());
    CPPUNIT_ASSERT(mse.memberNames[0] == "i");
    CPPUNIT_ASSERT(mse.memberNames[1] == "j");
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[0].type);
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[1].type);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44.0, mse.memberValues[0].scalar.value, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(55.0, mse.memberValues[1].scalar.value, 1e-10);
    mse = ms.elements[1];
    CPPUNIT_ASSERT_EQUAL(2, (int)mse.memberNames.size());
    CPPUNIT_ASSERT_EQUAL(2, (int)mse.memberValues.size());
    CPPUNIT_ASSERT(mse.memberNames[0] == "i");
    CPPUNIT_ASSERT(mse.memberNames[1] == "j");
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[0].type);
    CPPUNIT_ASSERT_EQUAL((int)CELL_SCALAR, mse.memberValues[1].type);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(66.0, mse.memberValues[0].scalar.value, 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(77.0, mse.memberValues[1].scalar.value, 1e-10);

    // check strings are as expected
    vector<MatlabString> *strings = mp->getStrings();
    CPPUNIT_ASSERT_EQUAL(1, (int)strings->size());
    CPPUNIT_ASSERT(strings->at(0).name == "string1");
    CPPUNIT_ASSERT(strings->at(0).value == "This is a single-quoted string");

    delete mp;
}

