/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "InstrumentParserZero.h"
#include "Logger.h"
#include "MaterialsManager.h"
#include "GlobalSettings.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

InstrumentParserZero::InstrumentParserZero(string filename) : InstrumentParser(filename)
{
}

InstrumentParserZero::~InstrumentParserZero()
{
}

Instrument *InstrumentParserZero::parse()
{
    logMessage(1, "InstrumentParserZero: attempting to parse %s", filename.c_str());

    instrument = new Instrument();
    if (!parseTextFile()) {
	delete instrument;
	instrument = NULL;
    }

    return instrument;
}

Plate *InstrumentParserZero::readPlate(istream &in)
{
    string name, matname;
    double thickness, tension, lx, ly, t60_0, t60_1000;
    int bc;

    in >> name >> matname >> thickness >> tension >> lx >> ly >> t60_0 >> t60_1000 >> bc;

    if (in.fail()) {
	logMessage(3, "InstrumentParserZero: error parsing plate definition in instrument file");
	return NULL;
    }
    
    Material *material = MaterialsManager::getInstance()->getMaterial(matname);
    if (material == NULL) {
	logMessage(3, "InstrumentParserZero: unrecognised material '%s' requested", matname.c_str());
	return NULL;
    }

    logMessage(3, "Creating plate %s, %s, %f, %f, %f, %f, %f, %f, %d", name.c_str(),
	       matname.c_str(), thickness, tension, lx, ly, t60_0, t60_1000, bc);

    Plate *pl = new Plate(name, material, thickness, tension, lx, ly, t60_0, t60_1000, bc);
    return pl;
}

Output *InstrumentParserZero::readOutput(istream &in)
{
    string compname;
    double x, y, pan;

    in >> compname >> x >> y >> pan;
    if (in.fail()) {
	logMessage(3, "InstrumentParserZero: error parsing output definition");
	return NULL;
    }
    Component *comp = instrument->getComponentByName(compname);
    if (comp == NULL) {
	logMessage(3, "InstrumentParserZero: unrecognised component name '%s'",
		   compname.c_str());
	return NULL;
    }

    Output *op = new Output(comp, pan, x, y, 0.0);
    logMessage(3, "Creating output %s, %f, %f, %f", compname.c_str(), x, y, pan);
    return op;
}

ConnectionZero *InstrumentParserZero::readConnection(istream &in)
{
    string platename1, platename2;
    double x1, y1, x2, y2, ls, nls, t60;

    in >> platename1 >> platename2 >> x1 >> y1 >> x2 >> y2 >> ls >> nls >> t60;

    if (in.fail()) {
	logMessage(3, "InstrumentParserZero: error parsing connection definition in instrument file");
	return NULL;
    }

    double extra;
    in >> extra;
    if (!in.eof()) {
	logMessage(1, "InstrumentParserZero: looks like a ZeroPt1 style connection, bailing out");
	return NULL;
    }

    Component *comp1 = instrument->getComponentByName(platename1);
    if (comp1 == NULL) {
	logMessage(3, "InstrumentParserZero: unrecognised component name '%s' in connection definition",
		   platename1.c_str());
	return NULL;
    }

    Component *comp2 = instrument->getComponentByName(platename2);
    if (comp2 == NULL) {
	logMessage(3, "InstrumentParserZero: unrecognised component name '%s' in connection definition",
		   platename2.c_str());
	return NULL;
    }

    ConnectionZero *conn = new ConnectionZero(comp1, comp2, x1, y1, 0.0, x2, y2, 0.0, ls,
					      nls, t60);

    logMessage(3, "Creating connection %s, %s, %f, %f, %f, %f, %f, %f, %f",
	       platename1.c_str(), platename2.c_str(), x1, y1, x2, y2, ls, nls, t60);

    return conn;
}

int InstrumentParserZero::handleItem(string type, istream &in)
{
    int i;
    int result = 1;

    if (type == "plate") {
	Plate *plate = readPlate(in);
	if (plate) {
	    instrument->addComponent(plate);
	}
	else {
	    result = 0;
	}
    }
    else if (type == "connection") {
	ConnectionZero *conn = readConnection(in);
	if (conn) {
	    for (i = 0; i < conns.size(); i++) {
		// don't add one that co-incides with existing one
		if (conns[i]->coincides(conn)) {
		    delete conn;
		    conn = NULL;
		    break;
		}
	    }
	    if (conn) {
		instrument->addConnection(conn);
		conns.push_back(conn);
	    }
	}
	else {
	    result = 0;
	}
    }
    else if (type == "output") {
	Output *op = readOutput(in);
	if (op) {
	    instrument->addOutput(op);
	}
	else {
	    result = 0;
	}
    }
    else if (type == "samplerate") {
	int sr;
	in >> sr;
	logMessage(1, "Setting sample rate to %d", sr);
	GlobalSettings::getInstance()->setSampleRate((double)sr);
    }
    else if ((type == "normalise_outs") || (type == "normalize_outs")) {
	int no;
	in >> no;
	if (no != 0) {
	    logMessage(3, "Enabling output normalisation");
	    GlobalSettings::getInstance()->setNormaliseOuts(true);
	}
    }
    else {
	logMessage(3, "InstrumentParserZero: unrecognised line '%s'", type.c_str());
	result = 0;
    }
    return result;
}
