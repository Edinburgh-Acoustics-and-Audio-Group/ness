/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "ScoreParserModal.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "ModalPlate.h"
#include "InputModalStrike.h"
#include "InputModalSine.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
using namespace std;

ScoreParserModal::ScoreParserModal(string filename) : ScoreParser(filename)
{
}

ScoreParserModal::~ScoreParserModal()
{
}

bool ScoreParserModal::parse(Instrument *instrument)
{
    durationOnly = false;
    tail = 0.0;
    bpm = 100.0;
    currtime = 0.0;
    lastStrikeLength = 0.0;
    this->instrument = instrument;
    return parseTextFile();
}

bool ScoreParserModal::parseStrike(istream &in)
{
    double duration, amplitude, width, x, y;

    in >> duration >> amplitude >> width >> x >> y;
    if (in.fail()) {
	logMessage(3, "ScoreParserModal: error parsing strike definition");
	return false;
    }

    if (currtime == 0.0) {
	// work out time for first strike
	currtime = 3.0 * width;
	if (currtime >= 0.8) currtime = 0.8;
    }
    else {
	currtime = currtime + lastStrikeLength;
    }

    double start = currtime - width;

    // update current start time
    //currtime += (duration * (60.0 / bpm));
    lastStrikeLength = duration * (60.0 / bpm);

    if (durationOnly) return true;

    Component *comp = instrument->getComponentByName("modalplate");
    if (comp == NULL) {
	logMessage(3, "ScoreParserModal: modal plate component not found");
	return false;
    }

    ModalPlate *mp = dynamic_cast<ModalPlate*>(comp);
    if (mp == NULL) {
	logMessage(3, "ScoreParserModal: component is not a modal plate");
	return false;
    }

    InputModalStrike *strike = new InputModalStrike(mp, x, y, start, width*2, amplitude);
    mp->addInput(strike);

    return true;
}

bool ScoreParserModal::parseSine(istream &in)
{
    double time, force, frequency, rampUpTime, steadyTime, rampDownTime, x, y;

    in >> time >> force >> frequency >> rampUpTime >> steadyTime >> rampDownTime
       >> x >> y;
    if (in.fail()) {
	logMessage(3, "ScoreParserModal: error parsing sine definition");
	return false;
    }

    // handle duration here - getDuration() will add the tail
    currtime = time + rampUpTime + steadyTime + rampDownTime;

    if (durationOnly) return true;

    Component *comp = instrument->getComponentByName("modalplate");
    if (comp == NULL) {
	logMessage(3, "ScoreParserModal: modal plate component not found");
	return false;
    }

    ModalPlate *mp = dynamic_cast<ModalPlate*>(comp);
    if (mp == NULL) {
	logMessage(3, "ScoreParserModal: component is not a modal plate");
	return false;
    }

    InputModalSine *sine = new InputModalSine(mp, time, force, frequency,
					      rampUpTime, steadyTime, rampDownTime,
					      x, y);
    mp->addInput(sine);

    return true;
}

double ScoreParserModal::getDuration()
{
    durationOnly = true;
    tail = 0;
    bpm = 100;
    currtime = 0;
    lastStrikeLength = 0.0;
    parseTextFile();
    return currtime + tail;
}

int ScoreParserModal::handleItem(string type, istream &in)
{
    int result = 1;

    if (type == "strike") {
	if (!parseStrike(in)) result = 0;
    }
    else if (type == "sine") {
	if (!parseSine(in)) result = 0;
    }
    else if (type == "tail") {
	in >> tail;
	logMessage(1, "Setting tail to %f", tail);
    }
    else if (type == "bpm") {
	in >> bpm;
	logMessage(1, "Setting bpm to %f", bpm);
    }

    return result;
}
