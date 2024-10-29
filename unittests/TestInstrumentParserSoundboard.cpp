/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInstrumentParserSoundboard.h"
#include "GlobalSettings.h"
#include "SoundBoard.h"
#include "StringWithFrets.h"

void TestInstrumentParserSoundboard::setUp()
{
}

void TestInstrumentParserSoundboard::tearDown()
{
}

void TestInstrumentParserSoundboard::testInstrumentParserSoundboard()
{
    // parse the file
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-soundboard.txt", "soundboard");

    // check the operation succeeded
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(48000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-12);

    // check components are as expected
    vector<Component *> *components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(5, (int)components->size());
    
    StringWithFrets *swf1 = dynamic_cast<StringWithFrets*>(components->at(0));
    CPPUNIT_ASSERT(swf1 != NULL);
    CPPUNIT_ASSERT(swf1->getName() == "string1");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.020632359246225, swf1->getRho(), 1e-8);

    StringWithFrets *swf2 = dynamic_cast<StringWithFrets*>(components->at(1));
    CPPUNIT_ASSERT(swf2 != NULL);
    CPPUNIT_ASSERT(swf2->getName() == "string2");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.020097540404999, swf2->getRho(), 1e-8);

    StringWithFrets *swf3 = dynamic_cast<StringWithFrets*>(components->at(2));
    CPPUNIT_ASSERT(swf3 != NULL);
    CPPUNIT_ASSERT(swf3->getName() == "string3");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.020278498218867, swf3->getRho(), 1e-8);

    StringWithFrets *swf4 = dynamic_cast<StringWithFrets*>(components->at(3));
    CPPUNIT_ASSERT(swf4 != NULL);
    CPPUNIT_ASSERT(swf4->getName() == "string4");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.020546881519205, swf4->getRho(), 1e-8);

    SoundBoard *soundBoard = dynamic_cast<SoundBoard*>(components->at(4));
    CPPUNIT_ASSERT(soundBoard != NULL);
    CPPUNIT_ASSERT(soundBoard->getName() == "soundboard");

    // check outputs are as expected
    vector<Output *> *outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(7, (int)outputs->size());

    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == swf3);
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == swf2);
    CPPUNIT_ASSERT(outputs->at(2)->getComponent() == swf2);
    CPPUNIT_ASSERT(outputs->at(3)->getComponent() == swf4);

    CPPUNIT_ASSERT(outputs->at(4)->getComponent() == soundBoard);
    CPPUNIT_ASSERT(outputs->at(5)->getComponent() == soundBoard);
    CPPUNIT_ASSERT(outputs->at(6)->getComponent() == soundBoard);

    delete instrument;
}

