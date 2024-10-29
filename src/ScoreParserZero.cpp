/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-16. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "ScoreParserZero.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "ModalPlate.h"
#include "InputModalStrike.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
using namespace std;

ScoreParserZero::ScoreParserZero(string filename) : ScoreParser(filename)
{
}

ScoreParserZero::~ScoreParserZero()
{
}

bool ScoreParserZero::parse(Instrument *instrument)
{
    durationOnly = false;
    this->instrument = instrument;
    return parseTextFile();
}

Input *ScoreParserZero::parseStrike(istream &in)
{
    double startTime, x, y, duration, amplitude;
    string compname;

    in >> startTime >> compname >> x >> y >> duration >> amplitude;
    if (in.fail()) {
	logMessage(3, "ScoreParserZero: error parsing strike definition");
	return NULL;
    }

    Component *comp = instrument->getComponentByName(compname);
    if (comp == NULL) {
	logMessage(3, "ScoreParserZero: unrecognised component name '%s' in strike",
		   compname.c_str());
	return NULL;
    }

    Input *strike;

    ModalPlate *mp = dynamic_cast<ModalPlate*>(comp);
    if (mp == NULL) {
	strike = new InputStrike(comp, x, y, 0.0, startTime, duration, amplitude);
    }
    else {
	strike = new InputModalStrike(mp, x, y, startTime, duration, amplitude);
    }
    comp->addInput(strike);
    return strike;
}

InputBow *ScoreParserZero::parseBow(istream &in)
{
    double startTime, x, y, duration, famplitude, vamplitude, friction, rampTime;
    string compname;

    in >> startTime >> compname >> x >> y >> famplitude >> vamplitude >> duration
       >> friction >> rampTime;
    if (in.fail()) {
	logMessage(3, "ScoreParserZero: error parsing bow definition");
	return NULL;
    }

    Component *comp = instrument->getComponentByName(compname);
    if (comp == NULL) {
	logMessage(3, "ScoreParserZero: unrecognised component name '%s' in bow",
		   compname.c_str());
	return NULL;
    }

    InputBow *bow = new InputBow(comp, x, y, 0.0, startTime, duration, famplitude,
				 vamplitude, friction, rampTime);
    comp->addInput(bow);
    return bow;
}

InputWav *ScoreParserZero::parseAudio(istream &in)
{
    double startTime, x, y, gain;
    string compname, filename;

    in >> filename >> startTime >> compname >> x >> y >> gain;
    if (in.fail()) {
	logMessage(3, "ScoreParserZero: error parsing audio definition");
	return NULL;
    }

    Component *comp = instrument->getComponentByName(compname);
    if (comp == NULL) {
	logMessage(3, "ScoreParserZero: unrecognised component name '%s' in audio",
		   compname.c_str());
	return NULL;
    }

    InputWav *wav = new InputWav(comp, x, y, 0.0, filename, startTime, gain);

    comp->addInput(wav);
    return wav;
}

double ScoreParserZero::getDuration()
{
    durationOnly = true;
    duration = -1.0;
    parseTextFile();
    return duration;
}

int ScoreParserZero::handleItem(string type, istream &in)
{
    int result = 1;

    if (durationOnly) {
	if (type == "duration") {
	    in >> duration;
	    logMessage(3, "Setting duration to %f", duration);
	    return 1;
	}
    }
    else {
	if (type == "strike") {
	    Input *strike = parseStrike(in);
	    if (!strike) {
		result = 0;
	    }
	}
	else if (type == "bow") {
	    InputBow *bow = parseBow(in);
	    if (!bow) {
		result = 0;
	    }
	}
	else if (type == "audio") {
	    InputWav *wav = parseAudio(in);
	    if (!wav) {
		result = 0;
	    }
	}
	else if (type == "highpass") {
	    string onoff;
	    in >> onoff;
	    if (onoff == "on") {
		logMessage(3, "Enabling high-pass filter");
		GlobalSettings::getInstance()->setHighPassOn(true);
	    }
	    else if (onoff == "off") {
		logMessage(3, "Disabling high-pass filter");
		GlobalSettings::getInstance()->setHighPassOn(false);
	    }
	    else {
		logMessage(3, "ScoreParserZero: unrecognised setting '%s' for high-pass filter",
			   onoff.c_str());
		result = 0;
	    }	    
	}
	else if (type == "duration") {
	    // already done
	}
	else {
	    logMessage(3, "ScoreParserZero: unrecognised line '%s' in score file",
		       type.c_str());
	    result = 0;
	}
    }

    return result;
}
