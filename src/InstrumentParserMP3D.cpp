/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "InstrumentParserMP3D.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "SettingsManager.h"
#include "AirboxIndexed.h"

#include "PlateEmbedded.h"
#include "Embedding.h"
#include "Output.h"
#include "OutputDifference.h"
#include "OutputPressure.h"

#include <cmath>
using namespace std;

InstrumentParserMP3D::InstrumentParserMP3D(string filename) : InstrumentParser(filename)
{
    airbox = NULL;
}

InstrumentParserMP3D::~InstrumentParserMP3D()
{
}

Instrument *InstrumentParserMP3D::parse()
{
    logMessage(1, "InstrumentParserMP3D: attempting to parse %s", filename.c_str());
    
    instrument = new Instrument();
    if (!parseTextFile()) {
	delete instrument;
	instrument = NULL;
    }

    return instrument;
}

double InstrumentParserMP3D::adjustPos(double x)
{
    // MP3D uses range -1...1 for output positions. We use 0...1 everywhere
    return ((x + 1.0) / 2.0);
}

int InstrumentParserMP3D::handleItem(string type, istream &in)
{
    int result = 1;

    if (type == "airbox") {
	double lx, ly, lz, c_a, rho_a;
	if (airbox != NULL) {
	    logMessage(3, "InstrumentParserMP3D: multiple airboxes not supported");
	    return 0;
	}
	in >> lx >> ly >> lz >> c_a >> rho_a;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: error parsing airbox definition");
	    return 0;
	}

	airbox = new AirboxIndexed("airbox", lx, ly, lz, c_a, rho_a);
	instrument->addComponent(airbox);
    }
    else if (type == "drumshell") {
	// a cylindrical drum shell
	string name;
	double cx, cy, bz, R, H_shell;
	if (airbox == NULL) {
	    logMessage(3, "InstrumentParserMP3D: airbox must come before drumshell in instrument definition");
	    return 0;
	}
	in >> name >> cx >> cy >> bz >> R >> H_shell;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: error parsing drum shell definition");
	    return 0;
	}
	airbox->addDrumShell(cx, cy, bz + (0.5 * airbox->getLZ()), R, H_shell);
    }
    else if (type == "bassdrum") {
	double H_shell, R;             // drum height and radius
	double rho, H, T, E, nu, T60, sig1;  // membrane parameters
	double lx, ly, lz, c_a, rho_a; // airbox
	
	if (airbox != NULL) {
	    logMessage(3, "InstrumentParserMP3D: cannot have airbox and bassdrum in same instrument");
	    return 0;
	}

	in >> lx >> ly >> lz >> c_a >> rho_a >> H_shell >> R >> rho >> H >> T >> E >> nu >> T60 >> sig1;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: error parsing bassdrum definition");
	    return 0;
	}

	airbox = new AirboxIndexed("airbox", lx, ly, lz, c_a, rho_a);
	instrument->addComponent(airbox);

	int NPHshell = (int)floor(H_shell / airbox->getQ());
	int pd1 = floor((H_shell + 0.5 * (lz - H_shell)) / airbox->getQ());
	int pu1 = pd1 + 1;
	int pd2 = pd1 - NPHshell;
	int pu2 = pd2 + 1;
	logMessage(1, "Drum shell height parameters: %d, %d, %d, %d, %d", NPHshell,
		   pd1, pu1, pd2, pu2);

	airbox->addDrumShell(pu2 - 1, pd1 - 1, R);
	
	double cztop = (((double)pd1) * airbox->getQ()) - (airbox->getLZ() / 2.0);
	PlateEmbedded *top = new PlateEmbedded("drumtop", nu, rho, E, H, T, R, R, T60, sig1,
					       0.0, 0.0, cztop, true, true);
	instrument->addComponent(top);

	Embedding *et = new Embedding(airbox, top);
	instrument->addConnection(et);

	double czbottom = (((double)pd2) * airbox->getQ()) - (airbox->getLZ() / 2.0);
	PlateEmbedded *bottom = new PlateEmbedded("drumbottom", nu, rho, E, H, T, R, R, T60,
						  sig1, 0.0, 0.0, czbottom, true, true);
	instrument->addComponent(bottom);

	Embedding *eb = new Embedding(airbox, bottom);
	instrument->addConnection(eb);
    }
    else if (type == "membrane") {
	string name;
	double R, cx, cy, cz, rho, H, T, E, nu, T60, sig1;

	in >> name >> R >> cx >> cy >> cz >> rho >> H >> T >> E >> nu >> T60 >> sig1;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: Error parsing membrane definition in instrument file");
	    return 0;
	}
	PlateEmbedded *p = new PlateEmbedded(name, nu, rho, E, H, T, R, 0.0, T60, sig1, cx,
					     cy, cz, true, true);
	instrument->addComponent(p);

	if (airbox) {
	    Embedding *e = new Embedding(airbox, p);
	    instrument->addConnection(e);
	}
    }
    else if (type == "plate") {
	string name;
	double Lx, Ly, cx, cy, cz, rho, H, E, nu, T60, sig1;

	in >> name >> Lx >> Ly >> cx >> cy >> cz >> rho >> H >> E >> nu >> T60 >> sig1;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: error parsing plate definition");
	    return 0;
	}

	PlateEmbedded *p = new PlateEmbedded(name, nu, rho, E, H, 0.0, Lx, Ly, T60, sig1, cx,
					     cy, cz, false, false);
	instrument->addComponent(p);

	if (airbox) {
	    Embedding *e = new Embedding(airbox, p);
	    instrument->addConnection(e);
	}
    }
    else if (type == "airbox_output") {
	double x, y, z;
	double pan;
	
	if (airbox == NULL) {
	    logMessage(3, "InstrumentParserMP3D: airbox must come before outputs in instrument file");
	    return 0;
	}

	in >> x >> y >> z;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: error parsing airbox output");
	    return 0;
	}
	in >> pan;
	if (in.fail()) {
	    // pan is optional. default is centred
	    pan = 0.0;
	}
	Output *op = new OutputPressure(airbox, pan, adjustPos(x), adjustPos(y), adjustPos(z));
	instrument->addOutput(op);
    }
    else if (type == "plate_output") {
	string name;
	double x, y;
	in >> name >> x >> y;
	if (in.fail()) {
	    logMessage(3, "InstrumentParserMP3D: error parsing plate output");
	    return 0;
	}
	Component *comp = instrument->getComponentByName(name);
	if (comp == NULL) {
	    logMessage(3, "InstrumentParserMP3D: unrecognised plate name '%s'", name.c_str());
	    return 0;
	}
	Output *op;
	PlateEmbedded *p = (PlateEmbedded *)comp;
	if (p->getMembrane()) {
	    op = new OutputPressure(comp, 0.0, adjustPos(x), adjustPos(y), 0.0);
	}
	else {
	    op = new OutputDifference(comp, 0.0, adjustPos(x), adjustPos(y), 0.0);
	}
	instrument->addOutput(op);
    }
    else if (type == "samplerate") {
	int sr;
	in >> sr;
	logMessage(3, "Setting sample rate to %d", sr);
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
	// unrecognised type
	logMessage(3, "InstrumentParserMP3D: unrecognised line %s", type.c_str());
	result = 0;
    }

    return result;
}

