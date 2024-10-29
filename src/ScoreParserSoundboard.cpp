/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "ScoreParserSoundboard.h"
#include "InputStrike.h"
#include "InputPluck.h"
#include "Component.h"
#include "Logger.h"

ScoreParserSoundboard::ScoreParserSoundboard(string filename) : ScoreParser(filename)
{
}

ScoreParserSoundboard::~ScoreParserSoundboard()
{
}

bool ScoreParserSoundboard::parse(Instrument *instrument)
{
    this->instrument = instrument;
    return parseTextFile();
}

int ScoreParserSoundboard::handleItem(string type, istream &in)
{
    int result = 1;
    if (type == "duration") {
	// already done, ignore
    }
    else if ((type == "strike") || (type == "pluck")) {
	string name;
	double startTime, duration, force, position;

	in >> name >> startTime >> duration >> force >> position;
	if (in.fail()) {
	    logMessage(3, "ScoreParserSoundboard: Error parsing %s line in score file",
		       type.c_str());
	    return 0;
	}

	Component *comp = instrument->getComponentByName(name);
	if (!comp) {	    
	    logMessage(3, "ScoreParserSoundboard: reference to non-existent string %s in score file",
		       name.c_str());
	    return 0;
	}

	Input *input;
	if (type == "strike") {
	    input = new InputStrike(comp, position, 0.0, 0.0, startTime, duration, force);
	}
	else {
	    input = new InputPluck(comp, position, 0.0, 0.0, startTime, duration, force);
	}
	comp->addInput(input);
    }
    else {
	logMessage(3, "ScoreParserSoundboard: unrecognised line '%s' in score file",
		   type.c_str());
	result = 0;
    }
    return result;
}

