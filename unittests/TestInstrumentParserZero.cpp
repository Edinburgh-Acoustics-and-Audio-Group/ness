/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInstrumentParserZero.h"
#include "InstrumentParser.h"
#include "GlobalSettings.h"
#include "Plate.h"
#include "ConnectionZero.h"
#include "Output.h"

void TestInstrumentParserZero::setUp()
{
}

void TestInstrumentParserZero::tearDown()
{
}

void TestInstrumentParserZero::testInstrumentParserZero()
{
    // parse the test file
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-zero.txt", "zero");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(32000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    vector<Component*> *components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(4, (int)components->size());

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

    // check connections are as expected
    vector<Connection*> *connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(2, (int)connections->size()); // one should have been removed for coinciding
    
    ConnectionZero *cz = dynamic_cast<ConnectionZero*>(connections->at(0));
    CPPUNIT_ASSERT(cz != NULL);

    cz = dynamic_cast<ConnectionZero*>(connections->at(1));
    CPPUNIT_ASSERT(cz != NULL);

    // check output is as expected
    vector<Output*> *outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == plat1);

    delete instrument;
}

