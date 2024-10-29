/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestScoreParserZero.h"

#include "InputStrike.h"
#include "InputBow.h"
#include "InputWav.h"
#include "GlobalSettings.h"

void TestScoreParserZero::setUp()
{
    instrument = new Instrument();
    comp1 = new DummyComponent2D("comp1", 10, 10);
    comp2 = new DummyComponent2D("comp2", 10, 10);
    instrument->addComponent(comp1);
    instrument->addComponent(comp2);
}

void TestScoreParserZero::tearDown()
{
    delete instrument; // instrument will delete the components
}

void TestScoreParserZero::testScoreParserZero()
{
    // get duration
    double duration = ScoreParser::getDuration("score-zero.txt", "zero");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, duration, 1e-12);
    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("score-zero.txt", instrument, "zero"));

    CPPUNIT_ASSERT(GlobalSettings::getInstance()->getHighPassOn());

    // check that we have a strike on comp1
    vector<Input*> *inputs = comp1->getInputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)inputs->size());

    InputStrike *strike = dynamic_cast<InputStrike*>(inputs->at(0));
    CPPUNIT_ASSERT(strike != NULL);

    // check that we have a bow and a wav on comp2
    inputs = comp2->getInputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)inputs->size());

    InputBow *bow = dynamic_cast<InputBow*>(inputs->at(0));
    CPPUNIT_ASSERT(bow != NULL);

    InputWav *wav = dynamic_cast<InputWav*>(inputs->at(1));
    CPPUNIT_ASSERT(wav != NULL);
}

