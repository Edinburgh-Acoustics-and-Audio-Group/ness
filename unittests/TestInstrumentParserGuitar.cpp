/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestInstrumentParserGuitar.h"
#include "GuitarString.h"
#include "GlobalSettings.h"
#include "ConnectionNet1.h"

void TestInstrumentParserGuitar::setUp()
{
}

void TestInstrumentParserGuitar::tearDown()
{
}

void TestInstrumentParserGuitar::testInstrumentParserGuitar()
{
    // parse the file
    Instrument *instr = InstrumentParser::parseInstrument("instrument-guitar.m", "guitar");

    // check the operation succeeded
    CPPUNIT_ASSERT(instr != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(48000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-12);

    // check components are as expected
    vector<Component *> *components = instr->getComponents();
    CPPUNIT_ASSERT_EQUAL(6, (int)components->size());

    GuitarString *gs = dynamic_cast<GuitarString*>(components->at(0));
    CPPUNIT_ASSERT(gs != NULL);
    CPPUNIT_ASSERT(gs->getName() == "string1");
    gs = dynamic_cast<GuitarString*>(components->at(1));
    CPPUNIT_ASSERT(gs != NULL);
    CPPUNIT_ASSERT(gs->getName() == "string2");
    gs = dynamic_cast<GuitarString*>(components->at(2));
    CPPUNIT_ASSERT(gs != NULL);
    CPPUNIT_ASSERT(gs->getName() == "string3");
    gs = dynamic_cast<GuitarString*>(components->at(3));
    CPPUNIT_ASSERT(gs != NULL);
    CPPUNIT_ASSERT(gs->getName() == "string4");
    gs = dynamic_cast<GuitarString*>(components->at(4));
    CPPUNIT_ASSERT(gs != NULL);
    CPPUNIT_ASSERT(gs->getName() == "string5");
    gs = dynamic_cast<GuitarString*>(components->at(5));
    CPPUNIT_ASSERT(gs != NULL);
    CPPUNIT_ASSERT(gs->getName() == "string6");

    // check outputs
    vector<Output*> *outputs = instr->getOutputs();
    CPPUNIT_ASSERT_EQUAL(6, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == components->at(0));
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == components->at(1));
    CPPUNIT_ASSERT(outputs->at(2)->getComponent() == components->at(2));
    CPPUNIT_ASSERT(outputs->at(3)->getComponent() == components->at(3));
    CPPUNIT_ASSERT(outputs->at(4)->getComponent() == components->at(4));
    CPPUNIT_ASSERT(outputs->at(5)->getComponent() == components->at(5));

    // check connections (net1 style)
    vector<Connection*> *connections = instr->getConnections();
    CPPUNIT_ASSERT_EQUAL(6, (int)connections->size());
    ConnectionNet1 *conn = dynamic_cast<ConnectionNet1*>(connections->at(0));
    CPPUNIT_ASSERT(conn != NULL);
    conn = dynamic_cast<ConnectionNet1*>(connections->at(1));
    CPPUNIT_ASSERT(conn != NULL);
    conn = dynamic_cast<ConnectionNet1*>(connections->at(2));
    CPPUNIT_ASSERT(conn != NULL);
    conn = dynamic_cast<ConnectionNet1*>(connections->at(3));
    CPPUNIT_ASSERT(conn != NULL);
    conn = dynamic_cast<ConnectionNet1*>(connections->at(4));
    CPPUNIT_ASSERT(conn != NULL);
    conn = dynamic_cast<ConnectionNet1*>(connections->at(5));
    CPPUNIT_ASSERT(conn != NULL);

    delete instr;
}
