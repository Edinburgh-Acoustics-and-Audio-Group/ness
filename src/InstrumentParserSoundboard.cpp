/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "InstrumentParserSoundboard.h"
#include "GlobalSettings.h"
#include "Logger.h"

InstrumentParserSoundboard::InstrumentParserSoundboard(string filename) : InstrumentParser(filename)
{
}

InstrumentParserSoundboard::~InstrumentParserSoundboard()
{
}

Instrument *InstrumentParserSoundboard::parse()
{
    instrument = new Instrument();
    if (!parseTextFile()) {
	delete instrument;
	instrument = NULL;
    }
    return instrument;
}

int InstrumentParserSoundboard::handleItem(string type, istream &in)
{
    int result = 1;
    int i;
    if (type == "samplerate") {
	int sr;
	in >> sr;
	logMessage(3, "Setting sample rate to %d", sr);
	GlobalSettings::getInstance()->setSampleRate((double)sr);
    }
    else if (type == "string") {
	StringWithFrets *str = readString(in);
	if (!str) {
	    result = 0;
	}
	else {
	    instrument->addComponent(str);
	    strings.push_back(str);
	}
    }
    else if (type == "plate") {
	SoundBoard *sb = readSoundBoard(in);
	if (!sb) {
	    result = 0;
	}
	else {
	    instrument->addComponent(sb);
	}
    }
    else if (type == "collision") {
	double K, alpha;
	int iter;

	in >> K >> alpha >> iter;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserSoundboard: error reading collision info");
	    result = 0;
	}
	else {
	    for (i = 0; i < strings.size(); i++) {
		strings[i]->setParams(K, alpha, iter);
	    }
	}
    }
    else if (type == "string_out") {
	Output *op = readStringOutput(in);
	if (!op) {
	    result = 0;
	}
	else {
	    instrument->addOutput(op);
	}
    }
    else if (type == "plate_out") {
	Output *op = readBoardOutput(in);
	if (!op) {
	    result = 0;
	}
	else {
	    instrument->addOutput(op);
	}
    }
    else {
	logMessage(3, "InstrumentParserSoundboard: unrecognised item '%s' in instrument file",
		   type.c_str());
	result = 0;
    }
    return result;
}

StringWithFrets *InstrumentParserSoundboard::readString(istream &in)
{
    string name;
    double L, rho, T, E, r, T60_0, T60_1000, xc1, yc1, xc2, yc2;
    int numfrets;
    double fretheight, backboardheight, backboardvar;

    in >> name >> L >> rho >> T >> E >> r >> T60_0 >> T60_1000 >> xc1 >> yc1 >> xc2
       >> yc2 >> numfrets >> fretheight >> backboardheight >> backboardvar;

    if (in.fail()) {
	logMessage(3, "InstrumentParserSoundboard: error parsing string definition");
	return NULL;
    }

    StringWithFrets *str = new StringWithFrets(name, L, rho, T, E, r, T60_0, T60_1000, xc1,
					       yc1, xc2, yc2, numfrets, fretheight,
					       backboardheight, backboardvar);
    return str;
}

SoundBoard *InstrumentParserSoundboard::readSoundBoard(istream &in)
{
    int i;
    double rho, H, E, nu, T, Lx, Ly, T60_0, T60_1000;

    in >> rho >> H >> E >> nu >> T >> Lx >> Ly >> T60_0 >> T60_1000;
    if (in.fail()) {
	logMessage(3, "InstrumentParserSoundboard: error parsing plate definition");
	return NULL;
    }

    vector<ComponentString*> *cstrings = new vector<ComponentString*>();
    for (i = 0; i < strings.size(); i++) {
	cstrings->push_back(strings[i]);
    }
    SoundBoard *board = new SoundBoard("soundboard", nu, rho, E, H, T, Lx, Ly, T60_0,
				       T60_1000, 1, cstrings);
    return board;
}

Output *InstrumentParserSoundboard::readStringOutput(istream &in)
{
    string name;
    double pos, pan;

    in >> name >> pos >> pan;
    if (in.fail()) {
	logMessage(3, "InstrumentParserSoundboard: error parsing string output definition");
	return NULL;
    }

    Component *comp = instrument->getComponentByName(name);
    if (!comp) {
	logMessage(3, "InstrumentParserSoundboard: unrecognised component name %s",
		   name.c_str());
	return NULL;
    }

    Output *op = new Output(comp, pan, pos, 0.0, 0.0);
    return op;
}

Output *InstrumentParserSoundboard::readBoardOutput(istream &in)
{
    string name;
    double x, y, pan;

    in >> x >> y >> pan;
    if (in.fail()) {
	logMessage(3, "InstrumentParserSoundboard: error parsing string output definition");
	return NULL;
    }

    Component *comp = instrument->getComponentByName("soundboard");
    if (!comp) {
	logMessage(3, "InstrumentParserSoundboard: soundboard must be defined before outputs");
	return NULL;
    }

    Output *op = new Output(comp, pan, x, y, 0.0);
    return op;
}
