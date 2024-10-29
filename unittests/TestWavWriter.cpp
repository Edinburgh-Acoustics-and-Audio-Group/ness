/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestWavWriter.h"
#include "WavWriter.h"
#include "Output.h"
#include "DummyComponent1D.h"
#include "GlobalSettings.h"

#include <unistd.h>

#include <vector>
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

static unsigned char stereoExpected[] = {
    'R', 'I', 'F', 'F', 0x34, 0xb1, 0x02, 0x00,
    'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
    16, 0, 0, 0, 1, 0, 2, 0,
    0x44, 0xac, 0x00, 0x00, 0x10, 0xb1, 0x02, 0x00,
    4, 0, 16, 0, 'd', 'a', 't', 'a',
    0x10, 0xb1, 0x02, 0x00,

    0xaa, 0x2a, 0x44, 0x04, 0x55, 0x15, 0x34, 0xf3,
    0x88, 0x08, 0x33, 0x33, 0xde, 0xdd, 0x45, 0xc4,
    0x00, 0x40, 0x22, 0x22
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

void TestWavWriter::setUp()
{
}

void TestWavWriter::tearDown()
{
}

void TestWavWriter::testWavWriter()
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

    vector<Output*> outputs;
    outputs.push_back(output1);
    outputs.push_back(output2);

    // try writing a mono file
    WavWriter *writer = new WavWriter("testwav.wav");
    writer->writeMonoWavFile(output1, 1.5);
    delete writer;

    // read it and check contents
    checkFileContents("testwav.wav", monoExpected, 54);

    // try writing a stereo file
    writer = new WavWriter("testwav.wav");
    writer->writeStereoMix(&outputs, 1.5);
    delete writer;

    // read it and check contents
    checkFileContents("testwav.wav", stereoExpected, 64);

    delete output1;
    delete comp;
    unlink("testwav.wav");
}

