/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "ScoreParserBrass.h"
#include "Logger.h"
#include "MatlabParser.h"
#include "InputValve.h"
#include "InputLips.h"
#include "GlobalSettings.h"

ScoreParserBrass::ScoreParserBrass(string filename) : ScoreParser(filename)
{

}

ScoreParserBrass::~ScoreParserBrass()
{
}

double ScoreParserBrass::getDuration()
{
    durationOnly = true;
    duration = -1.0;
    parseInternal();
    return duration;
}

bool ScoreParserBrass::parse(Instrument *instrument)
{
    durationOnly = false;
    this->instrument = instrument;
    return parseInternal();
}

#define PROCESS_LIP_ARRAY(nm, vec) \
    if (arr.name == nm) { \
        for (j = 0; j < (arr.width * arr.height); j++) { \
	    vec.push_back(arr.data[j]); \
        } \
    } \

#define PROCESS_VALVE_ARRAY(nm, vec) \
    if (arr.name == nm) { \
        int arrsz = arr.width * arr.height; \
	int arrpos = 0; \
        while (arrpos < arrsz) { \
	    for (j = 0; j < Nvalve; j++) { \
	        vec[j].push_back(arr.data[arrpos]); \
		vec[j].push_back(arr.data[arrpos + j + 1]); \
	    } \
            arrpos = arrpos + Nvalve + 1; \
	} \
    } \
	
bool ScoreParserBrass::parseInternal()
{
    int i, j, k;
    logMessage(1, "ScoreParserBrass: attempting to parse %s", filename.c_str());

    MatlabParser matlabParser(filename);
    if (!matlabParser.parse()) {
	return false;
    }

    // now look for the required values
    vector<MatlabScalar> *scalars = matlabParser.getScalars();
    vector<MatlabArray> *arrays = matlabParser.getArrays();

    // scalar T is duration
    for (i = 0; i < scalars->size(); i++) {
	if (scalars->at(i).name == "T") {
	    duration = scalars->at(i).value;
	}
	if (!durationOnly) {
	    if (scalars->at(i).name == "maxout") {
		// maximum output for normalising
		GlobalSettings::getInstance()->setMaxOut(scalars->at(i).value);
	    }
	}
    }
    if (durationOnly) return true;

    // vectors for lips: Sr, mu, sigma, H, w, pressure, lip_frequency, vibamp, vibfreq, tremamp, tremfreq, noiseamp
    vector<double> Sr, mu, sigma, H, w, pressure, lip_frequency, vibamp, vibfreq, tremamp, tremfreq, noiseamp;

    // get number of valves
    BrassInstrument *brass = dynamic_cast<BrassInstrument*>(instrument->getComponentByName("brass"));
    if (!brass) {
	logMessage(3, "Score parser error, brass instrument not found");
	return false;
    }
    int Nvalve = brass->getNumValves();

    // vectors for valves: valvevibfreq, valvevibamp, valveopening - for each valve
    vector<double> *valvevibfreq, *valvevibamp, *valveopening;

    valvevibfreq = new vector<double>[Nvalve];
    valvevibamp = new vector<double>[Nvalve];
    valveopening = new vector<double>[Nvalve];

    for (i = 0; i < arrays->size(); i++) {
	MatlabArray arr = arrays->at(i);
	PROCESS_LIP_ARRAY("Sr", Sr);
	PROCESS_LIP_ARRAY("mu", mu);
	PROCESS_LIP_ARRAY("sigma", sigma);
	PROCESS_LIP_ARRAY("H", H);
	PROCESS_LIP_ARRAY("w", w);
	PROCESS_LIP_ARRAY("pressure", pressure);
	PROCESS_LIP_ARRAY("lip_frequency", lip_frequency);
	PROCESS_LIP_ARRAY("vibamp", vibamp);
	PROCESS_LIP_ARRAY("vibfreq", vibfreq);
	PROCESS_LIP_ARRAY("tremamp", tremamp);
	PROCESS_LIP_ARRAY("tremfreq", tremfreq);
	PROCESS_LIP_ARRAY("noiseamp", noiseamp);
	PROCESS_VALVE_ARRAY("valvevibfreq", valvevibfreq);
	PROCESS_VALVE_ARRAY("valvevibamp", valvevibamp);
	PROCESS_VALVE_ARRAY("valveopening", valveopening);
    }

    InputLips *lips = new InputLips(brass, Sr, mu, sigma, H, w, pressure, lip_frequency, vibamp, vibfreq,
				    tremamp, tremfreq, noiseamp);
    brass->addInput(lips);

    for (i = 0; i < Nvalve; i++) {
	InputValve *valve = new InputValve(brass, i, valveopening[i], valvevibfreq[i], valvevibamp[i]);
	brass->addInput(valve);
    }

    delete[] valvevibfreq;
    delete[] valvevibamp;
    delete[] valveopening;

    return true;
}
