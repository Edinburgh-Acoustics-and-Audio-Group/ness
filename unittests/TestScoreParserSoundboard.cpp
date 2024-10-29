/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestScoreParserSoundboard.h"

#include "GlobalSettings.h"
#include "InputStrike.h"
#include "InputPluck.h"

void TestScoreParserSoundboard::setUp()
{
    instrument = new Instrument();
    comp1 = new DummyComponent1D("comp1", 10);
    comp2 = new DummyComponent1D("comp2", 10);
    instrument->addComponent(comp1);
    instrument->addComponent(comp2);
}

void TestScoreParserSoundboard::tearDown()
{
    delete instrument; // instrument will delete the components for us
}

void TestScoreParserSoundboard::testScoreParserSoundboard()
{
    // get duration
    double duration = ScoreParser::getDuration("score-soundboard.txt", "soundboard");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, duration, 1e-12);

    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("score-soundboard.txt", instrument, "soundboard"));

    // check that we have a strike and a pluck on component 1
    vector<Input*> *inputs = comp1->getInputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)inputs->size());

    InputStrike *strike = dynamic_cast<InputStrike*>(inputs->at(0));
    CPPUNIT_ASSERT(strike != NULL);
    InputPluck *pluck = dynamic_cast<InputPluck*>(inputs->at(1));
    CPPUNIT_ASSERT(pluck != NULL);

    // check that we have a strike on component 2
    inputs = comp2->getInputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)inputs->size());

    strike = dynamic_cast<InputStrike*>(inputs->at(0));
    CPPUNIT_ASSERT(strike != NULL);
}

