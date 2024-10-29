/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestScoreParserBrass.h"
#include "GlobalSettings.h"
#include "BrassInstrument.h"
#include "Instrument.h"
#include "InputLips.h"


void TestScoreParserBrass::setUp()
{
}

void TestScoreParserBrass::tearDown()
{
}

void TestScoreParserBrass::testScoreParserBrass()
{
    // setup instrument with brass component
    Instrument *instr = new Instrument();
    vector<double> vpos, vdl, vbl, bore;
    vpos.push_back(600.0);
    vdl.push_back(20.0);
    vbl.push_back(200.0);
    bore.push_back(0.0);
    bore.push_back(17.34);
    bore.push_back(1381.3);
    bore.push_back(127.0);
    BrassInstrument *brass = new BrassInstrument("brass", 20.0, 1, vpos, vdl, vbl, bore);
    instr->addComponent(brass);

    // get duration
    double duration = ScoreParser::getDuration("score-brass.m", "brass");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, duration, 1e-12);

    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("score-brass.m", instr, "brass"));

    delete instr;
}

