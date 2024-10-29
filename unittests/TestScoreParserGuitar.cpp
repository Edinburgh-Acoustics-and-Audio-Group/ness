/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "TestScoreParserGuitar.h"
#include "InstrumentParser.h"
#include "GlobalSettings.h"
#include "GuitarString.h"
#include "InputPluck.h"

#include <cstdlib>
using namespace std;

void TestScoreParserGuitar::setUp()
{
}

void TestScoreParserGuitar::tearDown()
{
}

void TestScoreParserGuitar::testScoreParserGuitar()
{
    srand(0); // score functions use rand, make them predictable

    // first create a guitar instrument by parsing the instrument file
    Instrument *instr = InstrumentParser::parseInstrument("instrument-guitar.m", "guitar");

    // get duration
    double duration = ScoreParser::getDuration("score-guitar.m", "guitar");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, duration, 1e-12);

    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("score-guitar.m", instr, "guitar"));

    // check the inputs on each string
    vector<Component*> *components = instr->getComponents();

    // check how many there are
    vector<Input*> *inputs = components->at(0)->getInputs();
    CPPUNIT_ASSERT_EQUAL(12, (int)inputs->size());
    inputs = components->at(1)->getInputs();
    CPPUNIT_ASSERT_EQUAL(6, (int)inputs->size());
    inputs = components->at(2)->getInputs();
    CPPUNIT_ASSERT_EQUAL(9, (int)inputs->size());
    inputs = components->at(3)->getInputs();
    CPPUNIT_ASSERT_EQUAL(8, (int)inputs->size());
    inputs = components->at(4)->getInputs();
    CPPUNIT_ASSERT_EQUAL(7, (int)inputs->size());
    inputs = components->at(5)->getInputs();
    CPPUNIT_ASSERT_EQUAL(9, (int)inputs->size());

    // check that they're all plucks
    int i, j;
    for (i = 0; i < components->size(); i++) {
	inputs = components->at(i)->getInputs();
	for (j = 0; j < inputs->size(); j++) {
	    InputPluck *pluck = dynamic_cast<InputPluck*>(inputs->at(j));
	    CPPUNIT_ASSERT(pluck != NULL);
	}
    }

    delete instr;
}
