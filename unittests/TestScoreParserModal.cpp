/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 */

#include "TestScoreParserModal.h"
#include "InstrumentParser.h"
#include "InputModalStrike.h"

#include "GlobalSettings.h"

void TestScoreParserModal::setUp()
{
}

void TestScoreParserModal::tearDown()
{
}

void TestScoreParserModal::testScoreParserModal()
{
    // parse the instrument file
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-modalplate.txt", "modal");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // get duration
    double duration = ScoreParser::getDuration("score-modalplate.txt", "modal");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.423, duration, 1e-10);

    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("score-modalplate.txt", instrument, "modal"));

    Component *mp = instrument->getComponents()->at(0);
    
    // check that we have the correct number of modal strikes
    vector<Input*> *inputs = mp->getInputs();
    CPPUNIT_ASSERT_EQUAL(13, (int)inputs->size());

    int i;
    for (i = 0; i < inputs->size(); i++) {
	InputModalStrike *ms = dynamic_cast<InputModalStrike*>(inputs->at(i));
	CPPUNIT_ASSERT(ms != NULL);
    }

    delete instrument;
}

