/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestMatlabFunction_StrumGenMulti.h"
#include "InstrumentParserGuitar.h"

#include <cstdlib>
using namespace std;

void TestMatlabFunction_StrumGenMulti::setUp()
{
}

void TestMatlabFunction_StrumGenMulti::tearDown()
{
}

static const double expectedResult[] = {
    2.000000000000000, 0.600000000000000, 0.500000000000000, 0.003000000000000, 1.500000000000000,
    6.000000000000000, 0.006803754343094, 0.783101268291055, 0.001056619844752, 1.059688006695215,
    5.000000000000000, 0.028232947158736, 0.751608219086942, 0.000967044551143, 1.053645918962381,
    4.000000000000000, 0.035555494216064, 0.808635192927269, 0.000995479410372, 1.025774184952385,
    3.000000000000000, 0.057295689455837, 0.802144145631299, 0.001090445945035, 1.083239013600740,
    2.000000000000000, 0.082714234559198, 0.834767508709229, 0.000928320511071, 1.021393775251412,
    1.000000000000000, 0.090326011432487, 0.758861883300758, 0.000927446315357, 1.060835350845398,
    1.000000000000000, 0.193133581785082, 0.784151103079389, 0.000925958089356, 0.921761760405154,
    2.000000000000000, 0.229978490360071, 0.754921104849745, 0.001002586478881, 1.067822446938521,
    3.000000000000000, 0.242252796651913, 0.767365058831575, 0.001027510453541, 1.004857438013357,
    4.000000000000000, 0.259871659739815, 0.875644003821371, 0.000958503356883, 1.054271539558783,
    5.000000000000000, 0.280534899584267, 0.843186213804030, 0.000980045724418, 1.078305890401037,
    6.000000000000000, 0.295666294920103, 0.776393335562383, 0.001061544904002, 1.083805294793009,
    6.000000000000000, 0.391395105524638, 0.871892332058350, 0.001005199070044, 0.917211169571248,
    5.000000000000000, 0.413844276919889, 0.826116308321299, 0.001078046520510, 0.969778587049702,
    4.000000000000000, 0.431283426415773, 0.723203687818350, 0.000991540347455, 0.912619167665308,
    3.000000000000000, 0.454765599083512, 0.875301461068588, 0.001080441614697, 1.070183957354251,
    2.000000000000000, 0.475333314987520, 0.806361654515547, 0.000975041395274, 1.052049747273349,
    1.000000000000000, 0.500250707282802, 0.826835801725665, 0.001006321286832, 0.907856068670683
};

void TestMatlabFunction_StrumGenMulti::testMatlabFunction_StrumGenMulti()
{
    // reset RNG to something predictable
    srand(0);

    InstrumentParserGuitar::setNumStrings(6);

    MatlabFunction_StrumGenMulti *func = new MatlabFunction_StrumGenMulti();
    MatlabCellContent exc, param, result, T, density, dur, amp, pluckdur, pos;
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

    T.type = CELL_ARRAY;
    T.array.width = 4;
    T.array.height = 1;
    T.array.data = new double[4];
    T.array.data[0] = 0.0;
    T.array.data[1] = 0.2;
    T.array.data[2] = 0.4;
    T.array.data[3] = 0.6;
    func->setParameter(1, T);

    density.type = CELL_ARRAY;
    density.array.width = 4;
    density.array.height = 1;
    density.array.data = new double[4];

    dur.type = CELL_ARRAY;
    dur.array.width = 4;
    dur.array.height = 1;
    dur.array.data = new double[4];

    amp.type = CELL_ARRAY;
    amp.array.width = 4;
    amp.array.height = 1;
    amp.array.data = new double[4];

    pluckdur.type = CELL_ARRAY;
    pluckdur.array.width = 4;
    pluckdur.array.height = 1;
    pluckdur.array.data = new double[4];

    pos.type = CELL_ARRAY;
    pos.array.width = 4;
    pos.array.height = 1;
    pos.array.data = new double[4];

    for (i = 0; i < 4; i++) {
	density.array.data[i] = 5.0;
	dur.array.data[i] = 0.1;
	amp.array.data[i] = 1.0;
	pluckdur.array.data[i] = 0.001;
	pos.array.data[i] = 0.8;
    }

    func->setParameter(2, density);
    func->setParameter(3, dur);

    param.type = CELL_SCALAR;
    param.scalar.value = 2.0;
    func->setParameter(4, param);

    func->setParameter(5, amp);

    param.scalar.value = 0.2;
    func->setParameter(6, param);
    func->setParameter(7, pluckdur);
    func->setParameter(8, param);
    func->setParameter(9, pos);
    func->setParameter(10, param);
    func->setParameter(11, param);

    CPPUNIT_ASSERT(func->execute(&result));
    CPPUNIT_ASSERT_EQUAL((int)CELL_ARRAY, result.type);
    CPPUNIT_ASSERT_EQUAL(5, result.array.width);
    CPPUNIT_ASSERT_EQUAL(19, result.array.height);
    
    for (i = 0; i < (19*5); i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedResult[i], result.array.data[i], 1e-10);
    }

    delete[] exc.array.data;
    delete[] T.array.data;
    delete[] density.array.data;
    delete[] dur.array.data;
    delete[] amp.array.data;
    delete[] pluckdur.array.data;
    delete[] pos.array.data;
    delete[] result.array.data;
    delete func;
}

