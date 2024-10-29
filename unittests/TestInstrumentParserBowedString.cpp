/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestInstrumentParserBowedString.h"
#include "BowedString.h"
#include "GlobalSettings.h"

void TestInstrumentParserBowedString::setUp()
{
}

void TestInstrumentParserBowedString::tearDown()
{
}

void TestInstrumentParserBowedString::testInstrumentParserBowedString()
{
    // parse the file
    Instrument *instr = InstrumentParser::parseInstrument("Nbowedstrings_instrumentfile.m", "bowedstring");

    // check the operation succeeded
    CPPUNIT_ASSERT(instr != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, GlobalSettings::getInstance()->getSampleRate(), 1e-12);

    // check components are as expected
    vector<Component *> *components = instr->getComponents();
    CPPUNIT_ASSERT_EQUAL(4, (int)components->size());

    BowedString *bs = dynamic_cast<BowedString*>(components->at(0));
    CPPUNIT_ASSERT(bs != NULL);
    CPPUNIT_ASSERT(bs->getName() == "string1");
    bs = dynamic_cast<BowedString*>(components->at(1));
    CPPUNIT_ASSERT(bs != NULL);
    CPPUNIT_ASSERT(bs->getName() == "string2");
    bs = dynamic_cast<BowedString*>(components->at(2));
    CPPUNIT_ASSERT(bs != NULL);
    CPPUNIT_ASSERT(bs->getName() == "string3");
    bs = dynamic_cast<BowedString*>(components->at(3));
    CPPUNIT_ASSERT(bs != NULL);
    CPPUNIT_ASSERT(bs->getName() == "string4");

    // check outputs
    vector<Output*> *outputs = instr->getOutputs();
    CPPUNIT_ASSERT_EQUAL(4, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == components->at(0));
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == components->at(1));
    CPPUNIT_ASSERT(outputs->at(2)->getComponent() == components->at(2));
    CPPUNIT_ASSERT(outputs->at(3)->getComponent() == components->at(3));

    // check there's no connections
    CPPUNIT_ASSERT_EQUAL(0, (int)instr->getConnections()->size());

    delete instr;
}
