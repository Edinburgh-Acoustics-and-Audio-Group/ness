/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestScoreParserBowedString.h"
#include "InstrumentParser.h"
#include "GlobalSettings.h"
#include "BowedString.h"

#include <cstdlib>
using namespace std;

void TestScoreParserBowedString::setUp()
{
}

void TestScoreParserBowedString::tearDown()
{
}

void TestScoreParserBowedString::testScoreParserBowedString()
{
    // first create a bowed string instrument by parsing the instrument file
    Instrument *instr = InstrumentParser::parseInstrument("Nbowedstrings_instrumentfile.m", "bowedstring");

    // get duration
    double duration = ScoreParser::getDuration("Nbowedstrings_scorefile.m", "bowedstring");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, duration, 1e-12);

    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("Nbowedstrings_scorefile.m", instr, "bowedstring"));

    // not much else we can test in here... bowed string scores don't produce any
    // actual Inputs, but just modify the BowedString components in a way that can't
    // easily be detected...

    delete instr;
}
