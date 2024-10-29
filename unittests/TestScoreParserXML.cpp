/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestScoreParserXML.h"

#include "InputStrike.h"
#include "InputBow.h"
#include "InputWav.h"
#include "InputPluck.h"

void TestScoreParserXML::setUp()
{
    instrument = new Instrument();
    comp1 = new DummyComponent2D("comp1", 10, 10);
    comp2 = new DummyComponent2D("comp2", 10, 10);
    instrument->addComponent(comp1);
    instrument->addComponent(comp2);
}

void TestScoreParserXML::tearDown()
{
    delete instrument; // instrument will delete the components
}

void TestScoreParserXML::testScoreParserXML()
{
    // get duration
    double duration = ScoreParser::getDuration("score.xml");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, duration, 1e-12);
    // parse the score file
    CPPUNIT_ASSERT(ScoreParser::parseScore("score.xml", instrument));


    // check that we have a strike and a wav on comp1
    vector<Input*> *inputs = comp1->getInputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)inputs->size());

    InputStrike *strike = dynamic_cast<InputStrike*>(inputs->at(0));
    CPPUNIT_ASSERT(strike != NULL);
    InputWav *wav = dynamic_cast<InputWav*>(inputs->at(1));
    CPPUNIT_ASSERT(wav != NULL);

    // check that we have a pluck and a bow on comp2
    inputs = comp2->getInputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)inputs->size());

    InputPluck *pluck = dynamic_cast<InputPluck*>(inputs->at(0));
    CPPUNIT_ASSERT(pluck != NULL);

    InputBow *bow = dynamic_cast<InputBow*>(inputs->at(1));
    CPPUNIT_ASSERT(bow != NULL);
}

