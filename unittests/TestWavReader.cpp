/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestWavReader.h"
#include "WavReader.h"

void TestWavReader::setUp()
{
}

void TestWavReader::tearDown()
{
}

void TestWavReader::testWavReader()
{
    // read a 16-bit mono WAV file
    WavReader *reader = new WavReader("sine16m.wav");
    double *data = reader->getValues();

    // check that it was read in
    CPPUNIT_ASSERT(data != NULL);

    // check size
    CPPUNIT_ASSERT_EQUAL(30*44100, reader->getSize());

    // check sample rate
    CPPUNIT_ASSERT_EQUAL(44100, reader->getSampleRate());

    // check values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, data[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1640.0, data[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3280.0, data[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4900.0, data[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6505.0, data[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(23412.0, data[1000000], 1e-12);

    // try resampling
    reader->resampleTo(32000);
    CPPUNIT_ASSERT_EQUAL(32000, reader->getSampleRate());
    CPPUNIT_ASSERT_EQUAL(30*32000, reader->getSize());

    data = reader->getValues();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, data[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2260.125, data[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4505.125, data[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.534698853734881, data[900000], 1e-12);

    delete reader;


    // read 8-bit mono WAV file
    reader = new WavReader("sine8m.wav");
    data = reader->getValues();

    // check it was read in
    CPPUNIT_ASSERT(data != NULL);

    // check size and sample rate
    CPPUNIT_ASSERT_EQUAL(44100, reader->getSampleRate());
    CPPUNIT_ASSERT_EQUAL(30*44100, reader->getSize());

    // check a few values
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, data[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(12.0, data[2], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(19.0, data[3], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(26.0, data[4], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(92.0, data[1000000], 1e-12);

    // try normalising
    reader->normalise(1.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, data[0], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0679611650485437, data[1], 1e-12);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.893203883495146, data[1000000], 1e-12);

    delete reader;
}

