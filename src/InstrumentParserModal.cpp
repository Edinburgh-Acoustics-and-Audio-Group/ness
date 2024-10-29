/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "InstrumentParserModal.h"
#include "Logger.h"
#include "GlobalSettings.h"

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

InstrumentParserModal::InstrumentParserModal(string filename)
    : InstrumentParser(filename)
{
}

InstrumentParserModal::~InstrumentParserModal()
{
}

Instrument *InstrumentParserModal::parse()
{
    logMessage(1, "InstrumentParserModal: attempting to parse %s", filename.c_str());

    instrument = new Instrument();
    if (!parseTextFile()) {
	delete instrument;
	instrument = NULL;
    }

    return instrument;
}

int InstrumentParserModal::handleItem(string type, istream &in)
{
    int result = 1;
    if (type == "plate") {
	ModalPlate *mp = readModalPlate(in);
	if (mp) {
	    instrument->addComponent(mp);
	}
	else {
	    result = 0;
	}
    }
    else if (type == "output") {
	OutputModal *op = readOutput(in);
	if (op) {
	    instrument->addOutput(op);
	}
	else {
	    result = 0;
	}
    }
    else if (type == "samplerate") {
	logMessage(3, "Warning: sample rate in modal plate instrument file will be ignored");
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
	logMessage(3, "InstrumentParserModal: unrecognised line '%s'", type.c_str());
	result = 0;
    }
    return result;
}

ModalPlate *InstrumentParserModal::readModalPlate(istream &in)
{
    int A;
    double Lx, Ly, h, nu, Young, rho, loss1 = -1.0, loss2 = -1.0;
    double fs;
    bool halfNyquist = false;
    int num;

    in >> num;
    if (in.fail()) {
	logMessage(3, "InstrumentParserModal: error parsing plate definition in instrument file");
	return NULL;
    }

    in >> loss1 >> loss2;
    if ((in.fail()) || (loss2 < 0.0)) {
	loss1 = -1.0;
	loss2 = -1.0;
    }

    switch (num) {
    case 1:
	A = 9;
	Lx = 0.22;
	Ly = 0.21;
	h = 0.0005;
	nu = 0.3;
	Young = 2e11;
	rho = 7860.0;
	fs = 20000.0;
	if (loss2 < 0.0) {
	    loss1 = 1.0;
	    loss2 = 0.1;
	}
	break;
    case 2:
	A = 9;
	Lx = 0.32;
	Ly = 0.31;
	h = 0.001;
	nu = 0.3;
	Young = 2e11;
	rho = 7860.0;
	fs = 20000.0;
	if (loss2 < 0.0) {
	    loss1 = 1.0;
	    loss2 = 0.1;
	}
	break;
    case 3: // sine demo plate
	A = 9;
	Lx = 0.2;
	Ly = 0.2;
	h = 0.0002;
	nu = 0.3;
	Young = 2e11;
	rho = 7860.0;
	fs = 40000.0;
	halfNyquist = true;
	if (loss2 < 0.0) {
	    loss1 = 0.2;
	    loss2 = 0.0;
	}
	break;
    default:
	logMessage(3, "Unrecognised modal plate number %d in instrument file", num);
	return NULL;
    }

    GlobalSettings::getInstance()->setSampleRate(fs);
    GlobalSettings::getInstance()->setResampleOuts(true);

    logMessage(3, "Creating modal plate %d: %d, %f, %f, %f, %f, %f, %f, %f, %f, %d",
	       num, A, Lx, Ly, h, nu, Young, rho, loss1, loss2, (int)halfNyquist);
    ModalPlate *mp = new ModalPlate("modalplate", A, Lx, Ly, h, nu, Young, rho,
				    loss1, loss2, halfNyquist);
    return mp;
}

OutputModal *InstrumentParserModal::readOutput(istream &in)
{
    string compname = "modalplate";
    double x, y, pan;

    in >> x >> y >> pan;
    if (in.fail()) {
	logMessage(3, "InstrumentParserModal: error parsing output definition");
	return NULL;
    }
    Component *comp = instrument->getComponentByName(compname);
    if (comp == NULL) {
	logMessage(3, "InstrumentParserModal: unrecognised component name '%s'",
		   compname.c_str());
	return NULL;
    }
    ModalPlate *mp = dynamic_cast<ModalPlate*>(comp);

    OutputModal *op = new OutputModal(mp, pan, x, y);
    logMessage(3, "Creating output %s, %f, %f, %f", compname.c_str(), x, y, pan);
    return op;
}
