/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "TestInstrumentParserXML.h"

#include "InstrumentParser.h"
#include "PlateEmbedded.h"
#include "Plate.h"
#include "StringWithFrets.h"
#include "SoundBoard.h"
#include "Airbox.h"
#include "Fretboard.h"
#include "BrassInstrument.h"
#include "GuitarString.h"
#include "BowedString.h"
#include "ModalPlate.h"

#include "ConnectionZero.h"
#include "ConnectionZeroPt1.h"
#include "Embedding.h"
#include "ConnectionNet1.h"

#include "OutputPressure.h"
#include "OutputDifference.h"
#include "OutputModal.h"

#include "GlobalSettings.h"


void TestInstrumentParserXML::setUp()
{
}

void TestInstrumentParserXML::tearDown()
{
}

void TestInstrumentParserXML::testInstrumentParserXML()
{
    /*============================================================
     *
     * MP3D test file
     *
     *============================================================*/
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-mp3d.xml");

    // check it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(32000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-12);

    // check components are as expected
    vector<Component *> *components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(4, (int)components->size());

    Airbox *airbox = dynamic_cast<Airbox*>(components->at(0));
    CPPUNIT_ASSERT(airbox != NULL);
    CPPUNIT_ASSERT(airbox->getName() == "airbox");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.32, airbox->getLX(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.32, airbox->getLY(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.37, airbox->getLZ(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.21, airbox->getRhoA(), 1e-12);

    PlateEmbedded *plat1 = dynamic_cast<PlateEmbedded*>(components->at(1));
    CPPUNIT_ASSERT(plat1 != NULL);
    CPPUNIT_ASSERT(plat1->getName() == "plat1");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.81, plat1->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.87, plat1->getLy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plat1->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plat1->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.22, plat1->getCz(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7800.0, plat1->getRho(), 1e-8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002, plat1->getThickness(), 1e-12);
    CPPUNIT_ASSERT(!plat1->getCircular());
    CPPUNIT_ASSERT(!plat1->getMembrane());

    PlateEmbedded *plat2 = dynamic_cast<PlateEmbedded*>(components->at(2));
    CPPUNIT_ASSERT(plat2 != NULL);
    CPPUNIT_ASSERT(plat2->getName() == "plat2");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.39, plat2->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.42, plat2->getLy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.1, plat2->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.1, plat2->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plat2->getCz(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7800.0, plat2->getRho(), 1e-8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002, plat2->getThickness(), 1e-12);
    CPPUNIT_ASSERT(!plat2->getCircular());
    CPPUNIT_ASSERT(!plat2->getMembrane());

    PlateEmbedded *plat3 = dynamic_cast<PlateEmbedded*>(components->at(3));
    CPPUNIT_ASSERT(plat3 != NULL);
    CPPUNIT_ASSERT(plat3->getName() == "plat3");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.65, plat3->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.61, plat3->getLy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1, plat3->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1, plat3->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.27, plat3->getCz(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7800.0, plat3->getRho(), 1e-8);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002, plat3->getThickness(), 1e-12);
    CPPUNIT_ASSERT(!plat3->getCircular());
    CPPUNIT_ASSERT(!plat3->getMembrane());

    // check outputs
    vector<Output*> *outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(14, (int)outputs->size());

    int i;
    for (i = 0; i < 8; i++) {
	OutputPressure *op = dynamic_cast<OutputPressure*>(outputs->at(i));
	CPPUNIT_ASSERT(op != NULL);
	CPPUNIT_ASSERT(op->getComponent() == airbox);
    }

    OutputDifference *od = dynamic_cast<OutputDifference*>(outputs->at(8));
    CPPUNIT_ASSERT(od != NULL);
    CPPUNIT_ASSERT(od->getComponent() == plat1);

    od = dynamic_cast<OutputDifference*>(outputs->at(9));
    CPPUNIT_ASSERT(od != NULL);
    CPPUNIT_ASSERT(od->getComponent() == plat1);

    od = dynamic_cast<OutputDifference*>(outputs->at(10));
    CPPUNIT_ASSERT(od != NULL);
    CPPUNIT_ASSERT(od->getComponent() == plat2);

    od = dynamic_cast<OutputDifference*>(outputs->at(11));
    CPPUNIT_ASSERT(od != NULL);
    CPPUNIT_ASSERT(od->getComponent() == plat2);

    od = dynamic_cast<OutputDifference*>(outputs->at(12));
    CPPUNIT_ASSERT(od != NULL);
    CPPUNIT_ASSERT(od->getComponent() == plat3);

    od = dynamic_cast<OutputDifference*>(outputs->at(13));
    CPPUNIT_ASSERT(od != NULL);
    CPPUNIT_ASSERT(od->getComponent() == plat3);

    // check connections
    vector<Connection*> *connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(3, (int)connections->size());
    for (i = 0; i < 3; i++) {
	Embedding *emb = dynamic_cast<Embedding*>(connections->at(i));
	CPPUNIT_ASSERT(emb != NULL);
    }

    delete instrument;

    /*============================================================
     *
     * Sound board test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-soundboard.xml");

    // check the operation succeeded
    CPPUNIT_ASSERT(instrument != NULL);

    // check components are as expected
    components = instrument->getComponents();
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
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(7, (int)outputs->size());

    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == swf3);
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == swf2);
    CPPUNIT_ASSERT(outputs->at(2)->getComponent() == swf2);
    CPPUNIT_ASSERT(outputs->at(3)->getComponent() == swf4);

    CPPUNIT_ASSERT(outputs->at(4)->getComponent() == soundBoard);
    CPPUNIT_ASSERT(outputs->at(5)->getComponent() == soundBoard);
    CPPUNIT_ASSERT(outputs->at(6)->getComponent() == soundBoard);

    delete instrument;

    /*============================================================
     *
     * Zero code test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-zero.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(32000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(4, (int)components->size());

    Plate *zplat1 = dynamic_cast<Plate*>(components->at(0));
    CPPUNIT_ASSERT(zplat1 != NULL);
    CPPUNIT_ASSERT(zplat1->getName() == "plat1");

    Plate *zplat2 = dynamic_cast<Plate*>(components->at(1));
    CPPUNIT_ASSERT(zplat2 != NULL);
    CPPUNIT_ASSERT(zplat2->getName() == "plat2");

    Plate *zplat3 = dynamic_cast<Plate*>(components->at(2));
    CPPUNIT_ASSERT(zplat3 != NULL);
    CPPUNIT_ASSERT(zplat3->getName() == "plat3");

    Plate *zplat4 = dynamic_cast<Plate*>(components->at(3));
    CPPUNIT_ASSERT(zplat4 != NULL);
    CPPUNIT_ASSERT(zplat4->getName() == "plat4");

    // check connections are as expected
    connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(2, (int)connections->size()); // one should have been removed for coinciding
    
    ConnectionZero *cz = dynamic_cast<ConnectionZero*>(connections->at(0));
    CPPUNIT_ASSERT(cz != NULL);

    cz = dynamic_cast<ConnectionZero*>(connections->at(1));
    CPPUNIT_ASSERT(cz != NULL);

    // check output is as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == zplat1);

    delete instrument;


    /*============================================================
     *
     * ZeroPt1 test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-zeropt1.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(96000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(5, (int)components->size());

    zplat1 = dynamic_cast<Plate*>(components->at(0));
    CPPUNIT_ASSERT(zplat1 != NULL);
    CPPUNIT_ASSERT(zplat1->getName() == "plat1");

    zplat2 = dynamic_cast<Plate*>(components->at(1));
    CPPUNIT_ASSERT(zplat2 != NULL);
    CPPUNIT_ASSERT(zplat2->getName() == "plat2");

    zplat3 = dynamic_cast<Plate*>(components->at(2));
    CPPUNIT_ASSERT(zplat3 != NULL);
    CPPUNIT_ASSERT(zplat3->getName() == "plat3");

    zplat4 = dynamic_cast<Plate*>(components->at(3));
    CPPUNIT_ASSERT(zplat4 != NULL);
    CPPUNIT_ASSERT(zplat4->getName() == "plat4");

    Plate *zplat5 = dynamic_cast<Plate*>(components->at(4));
    CPPUNIT_ASSERT(zplat5 != NULL);
    CPPUNIT_ASSERT(zplat5->getName() == "plat5");

    // check connections are as expected
    connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(5, (int)connections->size());
    
    for (i = 0; i < 5; i++) {
	ConnectionZeroPt1 *czp1 = dynamic_cast<ConnectionZeroPt1*>(connections->at(i));
	CPPUNIT_ASSERT(czp1 != NULL);
    }

    // check outputs are as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == zplat1);
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == zplat2);

    delete instrument;

    /*============================================================
     *
     * Fretboard test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-fb.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(1, (int)components->size());

    Fretboard *fb = dynamic_cast<Fretboard*>(components->at(0));
    CPPUNIT_ASSERT(fb != NULL);
    CPPUNIT_ASSERT(fb->getName() == "fretboard");

    // check no connections
    CPPUNIT_ASSERT_EQUAL(0, (int)instrument->getConnections()->size());

    // check outputs are as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == fb);
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == fb);
    
    delete instrument;


    /*============================================================
     *
     * Brass test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-trombonevalve.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(44100.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(1, (int)components->size());

    BrassInstrument *brass = dynamic_cast<BrassInstrument*>(components->at(0));
    CPPUNIT_ASSERT(brass != NULL);
    CPPUNIT_ASSERT(brass->getName() == "trombone");

    // check no connections
    CPPUNIT_ASSERT_EQUAL(0, (int)instrument->getConnections()->size());

    // check outputs are as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == brass);
    
    delete instrument;


    /*============================================================
     *
     * Guitar test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-guitar.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(48000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    components = instrument->getComponents();
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
    

    // check connections
    connections = instrument->getConnections();
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

    // check outputs are as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(6, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == components->at(0));
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == components->at(1));
    CPPUNIT_ASSERT(outputs->at(2)->getComponent() == components->at(2));
    CPPUNIT_ASSERT(outputs->at(3)->getComponent() == components->at(3));
    CPPUNIT_ASSERT(outputs->at(4)->getComponent() == components->at(4));
    CPPUNIT_ASSERT(outputs->at(5)->getComponent() == components->at(5));

    delete instrument;


    /*============================================================
     *
     * Single bowed string test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-bowedstring2.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(1, (int)components->size());
    BowedString *bs = dynamic_cast<BowedString*>(components->at(0));
    CPPUNIT_ASSERT(bs != NULL);
    CPPUNIT_ASSERT(bs->getName() == "string1");

    // check no connections
    CPPUNIT_ASSERT_EQUAL(0, (int)instrument->getConnections()->size());

    // check outputs are as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == components->at(0));

    delete instrument;


    /*============================================================
     *
     * Whole bowed string instrument test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-bowedstring1.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(4, (int)components->size());
    bs = dynamic_cast<BowedString*>(components->at(0));
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

    // check no connections
    CPPUNIT_ASSERT_EQUAL(0, (int)instrument->getConnections()->size());

    // check outputs are as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(2, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == components->at(0));
    CPPUNIT_ASSERT(outputs->at(1)->getComponent() == components->at(1));

    delete instrument;

    /*============================================================
     *
     * Modal plate test file
     *
     *============================================================*/
    instrument = InstrumentParser::parseInstrument("instrument-modalplate.xml");

    // check that it worked
    CPPUNIT_ASSERT(instrument != NULL);

    // check sample rate
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20000.0, GlobalSettings::getInstance()->getSampleRate(), 1e-10);

    // check components are as expected
    components = instrument->getComponents();
    CPPUNIT_ASSERT_EQUAL(1, (int)components->size());

    ModalPlate *mplat1 = dynamic_cast<ModalPlate*>(components->at(0));
    CPPUNIT_ASSERT(mplat1 != NULL);
    CPPUNIT_ASSERT(mplat1->getName() == "modalplate");

    // check connections are as expected
    connections = instrument->getConnections();
    CPPUNIT_ASSERT_EQUAL(0, (int)connections->size());

    // check output is as expected
    outputs = instrument->getOutputs();
    CPPUNIT_ASSERT_EQUAL(1, (int)outputs->size());
    CPPUNIT_ASSERT(outputs->at(0)->getComponent() == mplat1);
    
    delete instrument;

    GlobalSettings::getInstance()->setSampleRate(44100);
}

