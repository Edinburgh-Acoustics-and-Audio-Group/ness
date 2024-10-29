/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "InstrumentParserBowedString.h"
#include "BowedString.h"
#include "MatlabParser.h"
#include "Output.h"
#include "Logger.h"
#include "GlobalSettings.h"

#include <cstdio>
using namespace std;

InstrumentParserBowedString::InstrumentParserBowedString(string filename)
    : InstrumentParser(filename)
{
}

InstrumentParserBowedString::~InstrumentParserBowedString()
{
}

Instrument *InstrumentParserBowedString::parse()
{
    logMessage(1, "InstrumentParserBowedString: attempting to parse %s", filename.c_str());
    instrument = new Instrument();

    // parse the Matlab file
    MatlabParser matlabParser(filename);
    if (!matlabParser.parse()) {
	delete instrument;
	instrument = NULL;
	return NULL;
    }

    // extract require parameters
    // first look for instrument type
    int i, j;
    int Fs = -1, Nstrings = -1;
    int instr_numb = -1;

    double KB = 1e5, alphaB = 2.0, betaB = 20.0, lambdaB = 10.0, MB = 0.1;
    double KwF = 1e5, KuF = 1e3, alphaF = 2.2, betaF = 50.0, lambdaF = 20.0,
	MF = 0.05;

    string instrtype = "";
    vector<MatlabString> *strings = matlabParser.getStrings();
    vector<MatlabScalar> *scalars = matlabParser.getScalars();
    vector<MatlabStruct> *structs = matlabParser.getStructs();
    for (i = 0; i < strings->size(); i++) {
	if (strings->at(i).name == "instrument") {
	    logMessage(1, "Bowed string instrument type is %s", strings->at(i).value.c_str());
	    instrtype = strings->at(i).value;
	    break;
	}
    }

    // parse the scalars that are common to both modes
    for (i = 0; i < scalars->size(); i++) {
	if (scalars->at(i).name == "Fs") {
	    Fs = (int)scalars->at(i).value;
	    GlobalSettings::getInstance()->setSampleRate(Fs);
	}
	else if (scalars->at(i).name == "Nstrings") {
	    Nstrings = (int)scalars->at(i).value;
	}
	else if (scalars->at(i).name == "instr_numb") {
	    instr_numb = (int)scalars->at(i).value;
	}
	else if ((scalars->at(i).name == "normalize_outs") ||
		 (scalars->at(i).name == "normalise_outs")) {
	    int noval = (int)scalars->at(i).value;
	    if (noval != 0) {
		GlobalSettings::getInstance()->setNormaliseOuts(true);
	    }
	}
    }

    // find the bow and finger parameters if present
    for (i = 0; i < structs->size(); i++) {
	if (structs->at(i).name == "bow") {
	    MatlabStruct *bow = &structs->at(i);
	    MatlabStructElement &bowe = bow->elements[0];
	    for (j = 0; j < bowe.memberNames.size(); j++) {
		if (bowe.memberNames[j] == "Kw") {
		    KB = bowe.memberValues[j].scalar.value;
		}
		else if (bowe.memberNames[j] == "alpha") {
		    alphaB = bowe.memberValues[j].scalar.value;
		}
		else if (bowe.memberNames[j] == "beta") {
		    betaB = bowe.memberValues[j].scalar.value;
		}
		else if (bowe.memberNames[j] == "lambda") {
		    lambdaB = bowe.memberValues[j].scalar.value;
		}
		else if (bowe.memberNames[j] == "M") {
		    MB = bowe.memberValues[j].scalar.value;
		}
	    }
	}
	else if (structs->at(i).name == "fing") {
	    MatlabStruct *fing = &structs->at(i);
	    MatlabStructElement &finge = fing->elements[0];
	    for (j = 0; j < finge.memberNames.size(); j++) {
		if (finge.memberNames[j] == "Kw") {
		    KwF = finge.memberValues[j].scalar.value;
		}
		else if (finge.memberNames[j] == "Ku") {
		    KuF = finge.memberValues[j].scalar.value;
		}
		else if (finge.memberNames[j] == "alpha") {
		    alphaF = finge.memberValues[j].scalar.value;
		}
		else if (finge.memberNames[j] == "beta") {
		    betaF = finge.memberValues[j].scalar.value;
		}
		else if (finge.memberNames[j] == "lambda") {
		    lambdaF = finge.memberValues[j].scalar.value;
		}
		else if (finge.memberNames[j] == "M") {
		    MF = finge.memberValues[j].scalar.value;
		}
	    }
	}
    }

    if (instrtype != "") {
	// pre-defined instrument type
	// must have instr_numb defined as well
	if (instr_numb < 0) {
	    logMessage(3, "Error parsing bowed string instrument file: instr_numb must be specified when instrument is");
	    delete instrument;
	    return NULL;
	}

	// create the four strings
	for (i = 0; i < 4; i++) {
	    char namebuf[20];
	    sprintf(namebuf, "string%d", i+1);
	    BowedString *bs = new BowedString(namebuf, instrtype, instr_numb-1, i);
	    instrument->addComponent(bs);
	    bs->setBowParameters(KB, alphaB, betaB, lambdaB, MB);
	    bs->setFingerParameters(KwF, KuF, alphaF, betaF, lambdaF, MF);

	    // add an output for each string too
	    Output *output = new Output(bs, 0.0, 0.99);
	    instrument->addOutput(output);
	}
    }
    else {
	// manual instrument definition
	// find the strings struct
	MatlabStruct *strings = NULL;
	for (i = 0; i < structs->size(); i++) {
	    if (structs->at(i).name == "strings") {
		strings = &structs->at(i);
		break;
	    }
	}
	if (!strings) {
	    logMessage(3, "Error parsing bowed string instrument file: either instrument or strings must be defined");
	    delete instrument;
	    return NULL;
	}

	if (strings->elements.size() != Nstrings) {
	    logMessage(3, "Error parsing bowed string instrument file: Nstrings doesn't match number of strings specified");
	    delete instrument;
	    return NULL;
	}
	
	// now loop over strings and create them
	for (i = 0; i < Nstrings; i++) {
	    char namebuf[20];
	    double f0 = -1.0, rho = -1.0, rad = -1.0, E = -1.0;
	    double T60_0 = -1.0, T60_1000 = -1.0, L = -1.0;
	    sprintf(namebuf, "string%d", i+1);

	    for (j = 0; j < strings->elements[i].memberNames.size(); j++) {
		if (strings->elements[i].memberNames[j] == "f0") {
		    if (strings->elements[i].memberValues[j].type != CELL_SCALAR) {
			logMessage(3, "Error parsing bowed string instrument file: f0 must be a scalar value");
			delete instrument;
			return NULL;
		    }
		    f0 = strings->elements[i].memberValues[j].scalar.value;
		}
		else if (strings->elements[i].memberNames[j] == "rho") {
		    if (strings->elements[i].memberValues[j].type != CELL_SCALAR) {
			logMessage(3, "Error parsing bowed string instrument file: rho must be a scalar value");
			delete instrument;
			return NULL;
		    }
		    rho = strings->elements[i].memberValues[j].scalar.value;
		}
		else if (strings->elements[i].memberNames[j] == "rad") {
		    if (strings->elements[i].memberValues[j].type != CELL_SCALAR) {
			logMessage(3, "Error parsing bowed string instrument file: rad must be a scalar value");
			delete instrument;
			return NULL;
		    }
		    rad = strings->elements[i].memberValues[j].scalar.value;
		}
		else if (strings->elements[i].memberNames[j] == "E") {
		    if (strings->elements[i].memberValues[j].type != CELL_SCALAR) {
			logMessage(3, "Error parsing bowed string instrument file: E must be a scalar value");
			delete instrument;
			return NULL;
		    }
		    E = strings->elements[i].memberValues[j].scalar.value;
		}
		else if (strings->elements[i].memberNames[j] == "L") {
		    if (strings->elements[i].memberValues[j].type != CELL_SCALAR) {
			logMessage(3, "Error parsing bowed string instrument file: L must be a scalar value");
			delete instrument;
			return NULL;
		    }
		    L = strings->elements[i].memberValues[j].scalar.value;
		}
		else if (strings->elements[i].memberNames[j] == "T60") {
		    if ((strings->elements[i].memberValues[j].type != CELL_ARRAY) ||
			((strings->elements[i].memberValues[j].array.width*strings->elements[i].memberValues[j].array.height) != 2)) {
			logMessage(3, "Error parsing bowed string instrument file: T60 must be a two element array");
			delete instrument;
			return NULL;
		    }
		    T60_0 = strings->elements[i].memberValues[j].array.data[0];
		    T60_1000 = strings->elements[i].memberValues[j].array.data[1];
		}
	    }

	    if ((f0 < 0.0) || (rho < 0.0) || (rad < 0.0) || (E < 0.0) ||
		(T60_0 < 0.0) || (T60_1000 < 0.0) || (L < 0.0)) {
		logMessage(3, "Error parsing bowed string instrument file: string %d is missing a required value", i+1);
		delete instrument;
		return NULL;
	    }

	    BowedString *bs = new BowedString(namebuf, f0, rho, rad, E, T60_0, T60_1000, L);
	    instrument->addComponent(bs);
	    bs->setBowParameters(KB, alphaB, betaB, lambdaB, MB);
	    bs->setFingerParameters(KwF, KuF, alphaF, betaF, lambdaF, MF);

	    // add an output for each string too
	    Output *output = new Output(bs, 0.0, 0.99);
	    instrument->addOutput(output);
	}
    }

    return instrument;
}
