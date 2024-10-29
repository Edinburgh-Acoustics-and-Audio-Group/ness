/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "TestPlateEmbedded.h"

#include "GlobalSettings.h"

void TestPlateEmbedded::setUp()
{
    GlobalSettings::getInstance()->setPcgTolerance(1e-9);
    GlobalSettings::getInstance()->setLinear(false);

    airbox = new AirboxIndexed("airbox", 1.2, 1.2, 1.2, 340.0, 1.21);
    plate = new PlateEmbedded("plate", 0.33, 7800.0, 8e11, 0.002, 0.0, 0.8, 0.8,
			      4.0, 0.001, 0.0, 0.0, 0.0);
    embedding = new Embedding(airbox, plate);
}

void TestPlateEmbedded::tearDown()
{
    delete embedding;
    delete airbox;
    delete plate;
}

void TestPlateEmbedded::testPlateEmbedded()
{
    int i;

    // check the basic parameters are as expected
    CPPUNIT_ASSERT(plate->getName() == "plate");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCz(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, plate->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, plate->getLy(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.02046287500882e-6, plate->getA0(), 1e-15);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7800.0, plate->getRho(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002, plate->getThickness(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0242424242424242, plate->getH(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.60846414965699e-8, plate->getAlpha(), 1e-18);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.01929058101346, plate->getBowFactor(), 1e-12);

    CPPUNIT_ASSERT_EQUAL(32, plate->getNx());
    CPPUNIT_ASSERT_EQUAL(32, plate->getNy());

    // put in an impulse
    plate->getU1()[528] = 1.0;

    // run some timesteps
    for (i = 0; i < 100; i++) {
	airbox->runTimestep(i);
	plate->runTimestep(i);
	embedding->runTimestep(i);

	airbox->swapBuffers(i);
	plate->swapBuffers(i);
    }

    // now check values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0460445700310235, plate->getU()[527], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0273681976362305, plate->getU()[528], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0276586410667232, plate->getU()[529], 1e-10);

    delete plate;
    delete embedding;


    // create a fresh linear plate
    GlobalSettings::getInstance()->setLinear(true);
    plate = new PlateEmbedded("plate", 0.33, 7800.0, 8e11, 0.002, 0.0, 0.8, 0.8,
			      4.0, 0.001, 0.0, 0.0, 0.0);
    embedding = new Embedding(airbox, plate);

    // check the basic parameters are as expected
    CPPUNIT_ASSERT(plate->getName() == "plate");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCz(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, plate->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, plate->getLy(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.02046287500882e-6, plate->getA0(), 1e-15);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7800.0, plate->getRho(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002, plate->getThickness(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0242424242424242, plate->getH(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.60846414965699e-8, plate->getAlpha(), 1e-18);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.01929058101346, plate->getBowFactor(), 1e-12);

    CPPUNIT_ASSERT_EQUAL(32, plate->getNx());
    CPPUNIT_ASSERT_EQUAL(32, plate->getNy());

    // put in an impulse
    plate->getU1()[528] = 1.0;

    // run some timesteps
    for (i = 0; i < 100; i++) {
	airbox->runTimestep(i);
	plate->runTimestep(i);
	embedding->runTimestep(i);

	airbox->swapBuffers(i);
	plate->swapBuffers(i);
    }

    // now check values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.469861254636538, plate->getU()[527], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.558474815373403, plate->getU()[528], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.515191003870093, plate->getU()[529], 1e-10);

    delete plate;
    delete embedding;


    // Create fresh circular membrane
    GlobalSettings::getInstance()->setLinear(false);
    plate = new PlateEmbedded("membrane", 0.38, 1400.0, 3.5e9, 0.00025, 1050.0, 0.125, 0.125,
		      18.0, 0.001, 0.0, 0.0, 0.0, true, true);
    embedding = new Embedding(airbox, plate);

    // check the basic parameters are as expected
    CPPUNIT_ASSERT(plate->getName() == "membrane");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCz(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.25, plate->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.25, plate->getLy(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.97264894286588e-5, plate->getA0(), 1e-15);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1400.0, plate->getRho(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00025, plate->getThickness(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0036231884057971, plate->getH(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.000111905477690137, plate->getAlpha(), 1e-15);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.863658438041989, plate->getBowFactor(), 1e-12);

    CPPUNIT_ASSERT_EQUAL(70, plate->getNx());
    CPPUNIT_ASSERT_EQUAL(70, plate->getNy());

    // put in an impulse
    plate->getU1()[2485] = 1.0;

    // run some timesteps
    for (i = 0; i < 100; i++) {
	airbox->runTimestep(i);
	plate->runTimestep(i);
	embedding->runTimestep(i);

	airbox->swapBuffers(i);
	plate->swapBuffers(i);
    }

    // now check values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.043529619072308, plate->getU()[2484], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0471055318210777, plate->getU()[2485], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0464260043322405, plate->getU()[2486], 1e-10);

    delete plate;

    // Now create a plate with no embedding
    plate = new PlateEmbedded("plate", 0.33, 7800.0, 8e11, 0.002, 0.0, 0.8, 0.8,
			      4.0, 0.001, 0.0, 0.0, 0.0);
    
    CPPUNIT_ASSERT(plate->getName() == "plate");
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCy(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, plate->getCz(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, plate->getLx(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, plate->getLy(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.02046287500882e-6, plate->getA0(), 1e-15);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7800.0, plate->getRho(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.002, plate->getThickness(), 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0242424242424242, plate->getH(), 1e-12);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.60846414965699e-8, plate->getAlpha(), 1e-18);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.01929058101346, plate->getBowFactor(), 1e-12);

    CPPUNIT_ASSERT_EQUAL(32, plate->getNx());
    CPPUNIT_ASSERT_EQUAL(32, plate->getNy());

    // put in an impulse
    plate->getU1()[528] = 1.0;

    // run some timesteps
    for (i = 0; i < 100; i++) {
	plate->runTimestep(i);
	plate->swapBuffers(i);
    }

    // now check values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.025413785328511, plate->getU()[527], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.0285022092262957, plate->getU()[528], 1e-10);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.00376758469583735, plate->getU()[529], 1e-10);
}

