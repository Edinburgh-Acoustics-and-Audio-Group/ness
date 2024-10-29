/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestInstrumentParserModal.h"
#include "InstrumentParser.h"
#include "GlobalSettings.h"
#include "ModalPlate.h"

void TestInstrumentParserModal::setUp()
{
}

void TestInstrumentParserModal::tearDown()
{
}

void TestInstrumentParserModal::testInstrumentParserModal()
{
    // parse the test file
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-modalplate.txt", "modal");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    vector<Component*> *components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(1, (int)components->size());

    ModalPlate *plat1 = dynamic_cast<ModalPlate*>(components->at(0));
    CPPUNIT_ASSERT(plat1 != NULL);
    CPPUNIT_ASSERT(plat1->getName() == "modalplate");

    // check connections are as expected
    vector<Connection*> *connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(0, (int)connections->size());

    // check output is as expected
    vector<Output*> *outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == plat1);
    
    delete instrument;
}

