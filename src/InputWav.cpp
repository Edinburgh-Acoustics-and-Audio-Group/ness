/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "InputWav.h"
#include "WavReader.h"
#include "GlobalSettings.h"
#include "Logger.h"

#include <cstring>
#include <cstdlib>
using namespace std;

InputWav::InputWav(Component *comp, double x, double y, double z, string filename,
		   double startTime, double gain, int interpolated, int negated)
    : InputSample(comp, comp->getAlpha(), x, y, z, interpolated, negated)
{
    this->startTime = timeToTimestep(startTime);

    if (this->startTime < firstInputTimestep) firstInputTimestep = this->startTime;

    // load the WAV
    WavReader *wavFile = new WavReader(filename);
    if (!wavFile->getValues()) {
	// WAV loading failed, bail out
	// the WavReader already logged an appropriate error message
	exit(1);
    }

    // resample if necessary
    double sr = GlobalSettings::getInstance()->getSampleRate();
    wavFile->resampleTo((int)sr);

    // normalise to gain
    wavFile->normalise(gain);

    duration = wavFile->getSize();

    // copy to the data array. We can't just do 'data = wavFile->getValues()' because
    // then it will get double-freed
    data = new double[duration];
    memcpy(data, wavFile->getValues(), duration * sizeof(double));

    delete wavFile;
}

InputWav::~InputWav()
{
}

