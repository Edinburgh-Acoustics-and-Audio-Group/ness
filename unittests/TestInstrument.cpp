/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

// FIXME: this succeeds in isolation, but fails if run after all the other tests!!
// doesn't appear to be a duration or sample rate problem as we reset those at the start of the test, so this
// needs looked into

#include "TestInstrument.h"
#include "Instrument.h"
#include "Output.h"
#include "DummyComponent1D.h"
#include "GlobalSettings.h"

#include <unistd.h>

#include <cstdio>
using namespace std;

static unsigned char monoExpected[] = {
    'R', 'I', 'F', 'F', 0xac, 0x58, 0x01, 0x00,
    'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
    16, 0, 0, 0, 1, 0, 1, 0,
    0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,
    2, 0, 16, 0, 'd', 'a', 't', 'a',
    0x88, 0x58, 0x01, 0x00,

    0xaa, 0x2a, 0x55, 0x15, 0x88, 0x08, 0xde, 0xdd,
    0x00, 0x40
};

static unsigned char monoExpected2[] = {
    'R', 'I', 'F', 'F', 0xac, 0x58, 0x01, 0x00,
    'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
    16, 0, 0, 0, 1, 0, 1, 0,
    0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,
    2, 0, 16, 0, 'd', 'a', 't', 'a',
    0x88, 0x58, 0x01, 0x00,

    0x92, 0x04, 0x4a, 0xf2, 0xdb, 0x36, 0x00, 0xc0,
    0x92, 0x24
};

static unsigned char stereoExpected[] = {
    'R', 'I', 'F', 'F', 0x34, 0xb1, 0x02, 0x00,
    'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
    16, 0, 0, 0, 1, 0, 2, 0,
    0x44, 0xac, 0x00, 0x00, 0x10, 0xb1, 0x02, 0x00,
    4, 0, 16, 0, 'd', 'a', 't', 'a',
    0x10, 0xb1, 0x02, 0x00,

    0xd3, 0x1b, 0xc8, 0x02, 0xe9, 0x0d, 0xa7, 0xf7,
    0x90, 0x05, 0x64, 0x21, 0xbe, 0xe9, 0x0c, 0xd9,
    0xbd, 0x29, 0x42, 0x16
};

static void checkFileContents(char *filename, unsigned char *arr, int len)
{
    unsigned char *fc = new unsigned char[len];
    FILE *f = fopen(filename, "rb");
    CPPUNIT_ASSERT(f != NULL);
    fread(fc, 1, len, f);
    fclose(f);

    for (int i = 0; i < len; i++) {
	CPPUNIT_ASSERT_EQUAL(fc[i], arr[i]);
    }

    delete[] fc;
}

static double rawExpected1[] = { 1.0, 0.5, 0.2, -0.8, 1.5 };
static double rawExpected2[] = { 0.1, -0.3, 1.2, -1.4, 0.8 };

static void checkRawFileContents(char *filename, double *arr, int len)
{
    double *fc = new double[len];
    FILE *f = fopen(filename, "rb");
    CPPUNIT_ASSERT(f != NULL);
    fread(fc, 1, len * sizeof(double), f);
    fclose(f);

    for (int i = 0; i < len; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(fc[i], arr[i], 1e-10);
    }

    delete[] fc;
}

static void checkFileDoesntExist(char *filename)
{
    FILE *f = fopen(filename, "rb");
    CPPUNIT_ASSERT(f == NULL);
}

void TestInstrument::setUp()
{
}

void TestInstrument::tearDown()
{
}

void TestInstrument::testInstrument()
{
    GlobalSettings::getInstance()->setDuration(1.0);
    GlobalSettings::getInstance()->setSampleRate(44100.0);
    GlobalSettings::getInstance()->setHighPassOn(false);

    // set up test component and outputs
    DummyComponent1D *comp = new DummyComponent1D("comp", 10);
    Output *output1 = new Output(comp, -1.0, 0.0);
    Output *output2 = new Output(comp, 1.0, 0.0);
    double *data = output1->getData();
    data[0] = 1.0;
    data[1] = 0.5;
    data[2] = 0.2;
    data[3] = -0.8;
    data[4] = 1.5;
    data = output2->getData();
    data[0] = 0.1;
    data[1] = -0.3;
    data[2] = 1.2;
    data[3] = -1.4;
    data[4] = 0.8;

    // create instrument and add components and outputs
    Instrument *instr = new Instrument();
    instr->addComponent(comp);
    instr->addOutput(output1);
    instr->addOutput(output2);

    // test write outputs
    instr->saveOutputs("testoutput", true, true);

    // check individual WAV files
    checkFileContents("testoutput-comp-1.wav", monoExpected, 54);
    checkFileContents("testoutput-comp-2.wav", monoExpected2, 54);

    // check stereo mix
    checkFileContents("testoutput-mix.wav", stereoExpected, 64);

    // check raw files
    checkRawFileContents("testoutput-comp-1.f64", rawExpected1, 5);
    checkRawFileContents("testoutput-comp-2.f64", rawExpected2, 5);

    // delete outputs
    unlink("testoutput-comp-1.wav");
    unlink("testoutput-comp-2.wav");
    unlink("testoutput-mix.wav");
    unlink("testoutput-comp-1.f64");
    unlink("testoutput-comp-2.f64");

    // test again without raws or individuals
    instr->saveOutputs("testoutput", false, false);

    // check stereo mix
    checkFileContents("testoutput-mix.wav", stereoExpected, 64);

    // check that the others don't exist
    checkFileDoesntExist("testoutput-comp-1.wav");
    checkFileDoesntExist("testoutput-comp-2.wav");
    checkFileDoesntExist("testoutput-comp-1.f64");
    checkFileDoesntExist("testoutput-comp-2.f64");

    unlink("testoutput-mix.wav");

    // instrument will delete the other objects
    delete instr;
}

