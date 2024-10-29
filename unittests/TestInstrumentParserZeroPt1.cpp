/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInstrumentParserZeroPt1.h"
#include "GlobalSettings.h"
#include "Plate.h"
#include "ConnectionZeroPt1.h"

void TestInstrumentParserZeroPt1::setUp()
{
}

void TestInstrumentParserZeroPt1::tearDown()
{
}

void TestInstrumentParserZeroPt1::testInstrumentParserZeroPt1()
{
    // parse the test file
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-zeropt1.txt", "zeroPt1");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(96000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    vector<Component*> *components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(5, (int)components->size());

    Plate *plat1 = dynamic_cast<Plate*>(components->at(0));
    CPPUNIT_ASSERT(plat1 != NULL);
    CPPUNIT_ASSERT(plat1->getName() == "plat1");

    Plate *plat2 = dynamic_cast<Plate*>(components->at(1));
    CPPUNIT_ASSERT(plat2 != NULL);
    CPPUNIT_ASSERT(plat2->getName() == "plat2");

    Plate *plat3 = dynamic_cast<Plate*>(components->at(2));
    CPPUNIT_ASSERT(plat3 != NULL);
    CPPUNIT_ASSERT(plat3->getName() == "plat3");

    Plate *plat4 = dynamic_cast<Plate*>(components->at(3));
    CPPUNIT_ASSERT(plat4 != NULL);
    CPPUNIT_ASSERT(plat4->getName() == "plat4");

    Plate *plat5 = dynamic_cast<Plate*>(components->at(4));
    CPPUNIT_ASSERT(plat5 != NULL);
    CPPUNIT_ASSERT(plat5->getName() == "plat5");

    // check connections are as expected
    vector<Connection*> *connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(5, (int)connections->size());
    
    int i;
    for (i = 0; i < 5; i++) {
	ConnectionZeroPt1 *cz = dynamic_cast<ConnectionZeroPt1*>(connections->at(i));
	CPPUNIT_ASSERT(cz != NULL);
    }

    // check outputs are as expected
    vector<Output*> *outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == plat1);
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == plat2);

    delete instrument;
}

