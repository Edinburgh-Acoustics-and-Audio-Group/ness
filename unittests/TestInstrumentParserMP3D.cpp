/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestInstrumentParserMP3D.h"

#include "GlobalSettings.h"
#include "InstrumentParser.h"
#include "Airbox.h"
#include "PlateEmbedded.h"
#include "OutputPressure.h"
#include "OutputDifference.h"
#include "Embedding.h"

void TestInstrumentParserMP3D::setUp()
{
}

void TestInstrumentParserMP3D::tearDown()
{
}

void TestInstrumentParserMP3D::testInstrumentParserMP3D()
{
    // parse the file
    Instrument *instrument = InstrumentParser::parseInstrument("instrument-mp3d.txt", "mp3d");

    // check the operation succeeded
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
}

