/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 */

#include "ScoreParserBowedString.h"
#include "Logger.h"
#include "MatlabParser.h"
#include "GlobalSettings.h"
#include "BowedString.h"

#include <cstdio>
using namespace std;

ScoreParserBowedString::ScoreParserBowedString(string filename) : ScoreParser(filename)
{
}

ScoreParserBowedString::~ScoreParserBowedString()
{
}

double ScoreParserBowedString::getDuration()
{
    durationOnly = true;
    duration = -1.0;
    parseInternal();
    return duration;
}

bool ScoreParserBowedString::parse(Instrument *instrument)
{
    durationOnly = false;
    this->instrument = instrument;
    return parseInternal();
}

bool ScoreParserBowedString::parseInternal()
{
    int i, j, k, l;
    bool foundbowgest = false;

    logMessage(1, "ScoreParserBowedString:: attempting to parse %s", filename.c_str());

    MatlabParser matlabParser(filename);
    if (!matlabParser.parse()) {
	return false;
    }

    // find the duration
    vector<MatlabScalar> *scalars = matlabParser.getScalars();
    vector<MatlabStruct> *structs = matlabParser.getStructs();

    for (i = 0; i < scalars->size(); i++) {
	if (scalars->at(i).name == "Tf") {
	    duration = scalars->at(i).value;
	}
    }
    if (durationOnly) return true;

    // parse the bowgest and finggest structures
    vector<double> bowtime, bowpos, bowforce_w, bowforce_u;
    vector<double> fingtime, fingpos, fingforce_w, fingforce_u, vibrato;

    for (i = 0; i < structs->size(); i++) {
	if (structs->at(i).name == "bowgest") {
	    foundbowgest = true;

	    // found it. loop over the bows
	    for (j = 0; j < structs->at(i).elements.size(); j++) {
		MatlabStructElement &elem = structs->at(i).elements[j];

		bowtime.clear();
		bowpos.clear();
		bowforce_w.clear();
		bowforce_u.clear();

		double w0 = -1000000000.0, vw0 = -1000000000.0;
		double u0 = -1000000000.0, vu0 = -1000000000.0;
		int stringnum = -1;
		
		for (k = 0; k < elem.memberNames.size(); k++) {
		    if (elem.memberNames[k] == "stringnumber") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: stringnumber must be a scalar");
			    return false;
			}
			stringnum = (int)elem.memberValues[k].scalar.value;
		    }
		    else if (elem.memberNames[k] == "w0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: w0 must be a scalar");
			    return false;
			}
			w0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "vw0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: vw0 must be a scalar");
			    return false;
			}
			vw0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "u0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: u0 must be a scalar");
			    return false;
			}
			u0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "vu0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: vu0 must be a scalar");
			    return false;
			}
			vu0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "times") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: times must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    bowtime.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "pos") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: pos must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    bowpos.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "force_w") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: force_w must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    bowforce_w.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "force_u") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: force_u must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    bowforce_u.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else {
			logMessage(3, "Warning: unrecognised structure element %s in bowgest", elem.memberNames[k].c_str());
		    }
		}

		// sanity check the values
		if ((stringnum < 0) || (w0 < -1000000.0) || (vw0 < -1000000.0) || (u0 < -1000000.0) ||
		    (vu0 < -1000000.0) || (bowtime.size() == 0)) {
		    logMessage(5, "Error parsing bowed string score file: bowgest is missing a required value - %d %f %f %f %f %d", stringnum, w0, vw0, u0, vu0, bowtime.size());
		    return false;
		}

		if ((bowtime.size() != bowpos.size()) || (bowforce_w.size() != bowforce_u.size()) ||
		    (bowtime.size() != bowforce_w.size())) {
		    logMessage(5, "Error parsing bowed string score file: bowgest times, pos, force_w and force_u must be of equal length - %d %d %d %d", bowtime.size(), bowpos.size(), bowforce_w.size(), bowforce_u.size());
		    return false;
		}

		// find the string
		char namebuf[20];
		sprintf(namebuf, "string%d", stringnum);
		Component *comp = instrument->getComponentByName(namebuf);
		if (!comp) {
		    logMessage(5, "Error parsing bowed string score file: invalid string number %d", stringnum);
		    return false;
		}

		BowedString *bs = dynamic_cast<BowedString*>(comp);
		if (bs == NULL) {
		    logMessage(5, "Error parsing bowed string score file: string %d is not a BowedString!", stringnum);
		    return false;
		}

		// actually add the bow to the string
		bs->addBow(w0, vw0, u0, vu0, &bowtime, &bowpos, &bowforce_w, &bowforce_u);
	    }
	}
	else if (structs->at(i).name == "finggest") {
	    foundbowgest = true;

	    // found it. loop over the bows
	    for (j = 0; j < structs->at(i).elements.size(); j++) {
		MatlabStructElement &elem = structs->at(i).elements[j];

		fingtime.clear();
		fingpos.clear();
		fingforce_w.clear();
		fingforce_u.clear();
		vibrato.clear();

		double w0 = -1000000000.0, vw0 = -1000000000.0;
		double u0 = -1000000000.0, vu0 = -1000000000.0;
		int stringnum = -1;
		
		for (k = 0; k < elem.memberNames.size(); k++) {
		    if (elem.memberNames[k] == "stringnumber") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: stringnumber must be a scalar");
			    return false;
			}
			stringnum = (int)elem.memberValues[k].scalar.value;
		    }
		    else if (elem.memberNames[k] == "w0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: w0 must be a scalar");
			    return false;
			}
			w0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "vw0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: vw0 must be a scalar");
			    return false;
			}
			vw0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "u0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: u0 must be a scalar");
			    return false;
			}
			u0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "vu0") {
			if (elem.memberValues[k].type != CELL_SCALAR) {
			    logMessage(5, "Error parsing bowed string score file: vu0 must be a scalar");
			    return false;
			}
			vu0 = elem.memberValues[k].scalar.value;

		    }
		    else if (elem.memberNames[k] == "times") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: times must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    fingtime.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "pos") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: pos must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    fingpos.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "force_w") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: force_w must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    fingforce_w.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "force_u") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: force_u must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    fingforce_u.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else if (elem.memberNames[k] == "vibrato") {
			if (elem.memberValues[k].type != CELL_ARRAY) {
			    logMessage(5, "Error parsing bowed string score file: vibrato must be an array");
			    return false;
			}
			for (l = 0; l < (elem.memberValues[k].array.width*elem.memberValues[k].array.height); l++) {
			    vibrato.push_back(elem.memberValues[k].array.data[l]);
			}
		    }
		    else {
			logMessage(3, "Warning: unrecognised structure element %s in bowgest", elem.memberNames[k].c_str());
		    }
		}

		// sanity check the values
		if ((stringnum < 0) || (w0 < -100000.0) || (vw0 < -1000000.0) || (u0 < -1000000.0) ||
		    (vu0 < -1000000.0) || (fingtime.size() == 0)) {
		    logMessage(5, "Error parsing bowed string score file: finggest is missing a required value - %d %f %f %f %f %d", stringnum, w0, vw0, u0, vu0, fingtime.size());
		    return false;
		}

		if ((fingtime.size() != fingpos.size()) || (fingforce_w.size() != fingforce_u.size()) ||
		    (fingtime.size() != fingforce_w.size())) {
		    logMessage(5, "Error parsing bowed string score file: finggest times, pos, force_w and force_u must be of equal length - %d %d %d %d", fingtime.size(), fingpos.size(), fingforce_w.size(), fingforce_u.size());
		    return false;
		}

		if ((vibrato.size() % 5) != 0) {
		    logMessage(5, "Error parsing bowed string score file: vibrato array must have 5 values per vibrato gesture");
		    return false;
		}

		// find the string
		char namebuf[20];
		sprintf(namebuf, "string%d", stringnum);
		Component *comp = instrument->getComponentByName(namebuf);
		if (!comp) {
		    logMessage(5, "Error parsing bowed string score file: invalid string number %d", stringnum);
		    return false;
		}

		BowedString *bs = dynamic_cast<BowedString*>(comp);
		if (bs == NULL) {
		    logMessage(5, "Error parsing bowed string score file: string %d is not a BowedString!", stringnum);
		    return false;
		}

		// actually add the finger to the string
		bs->addFinger(w0, vw0, u0, vu0, &fingtime, &fingpos, &fingforce_w, &fingforce_u, &vibrato);
	    }
	}
    }
    return foundbowgest;
}
