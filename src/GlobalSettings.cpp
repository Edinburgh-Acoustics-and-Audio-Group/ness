/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2015. All rights reserved.
 */

#include "GlobalSettings.h"
#include "Logger.h"

GlobalSettings::GlobalSettings()
{
    logMessage(1, "GlobalSettings initialising to defaults");

    // sensible default values for everything
    sampleRate = 44100.0;
    duration = 1.0;
    k = 1.0 / sampleRate;

    energyOn = false;
    strikesOn = true;
    bowingOn = true;
    highPassOn = false;

    symmetricSolve = false;
    pcgTolerance = 0.000001;
    iterinv = -1;
    testStrike = false;
    noRecalcQ = false;
    lossMode = 1;

    fixpar = 1.0;

    linear = false;

    maxThreads = 1024;

    pcgMaxIterations = 500;
    interpolateInputs = false;
    interpolateOutputs = false;
    cuda2dBlockW = 16;
    cuda2dBlockH = 16;
    cuda3dBlockW = 8;
    cuda3dBlockH = 8;
    cuda3dBlockD = 8;
    negateInputs = false;
    logState = false;
    logMatrices = false;
    gpuEnable = true;

    estimate = false;

    avx = false;

    maxOut = 1.0;

    impulse = false;

    normaliseOuts = false;
    resampleOuts = false;
}

GlobalSettings::~GlobalSettings()
{
}

GlobalSettings *GlobalSettings::instance = NULL;
