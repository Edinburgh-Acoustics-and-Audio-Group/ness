/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInstrumentParserBrass.h"
#include "GlobalSettings.h"
#include "BrassInstrument.h"

void TestInstrumentParserBrass::setUp()
{
}

void TestInstrumentParserBrass::tearDown()
{
}

void TestInstrumentParserBrass::testInstrumentParserBrass()
{
    // parse the file
    Instrument *instr = InstrumentParser::parseInstrument("instrument-brass.m", "brass");

    // check the operation succeeded
    CPPUNIT_ASSERT(instr != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, GlobalSettings::getInstance()->getSampleRate(), 1e-12);

    // check components are as expected
    vector<Component *> *components = instr->getComponents();
    CPPUNIT_ASSERT_EQUAL(1, (int)components->size());
    
    BrassInstrument *brass = dynamic_cast<BrassInstrument*>(components->at(0));
    CPPUNIT_ASSERT(brass != NULL);
    CPPUNIT_ASSERT(brass->getName() == "brass");

    // check outputs
    vector<Output*> *outputs = instr->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == brass);

    // check there's no connections
    CPPUNIT_ASSERT_EQUAL(0, (int)instr->getConnections()->size());

    delete instr;
}

