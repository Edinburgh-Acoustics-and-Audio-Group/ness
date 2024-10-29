/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "ScoreParserGuitar.h"
#include "Logger.h"
#include "MatlabParser.h"
#include "InputPluck.h"
#include "InputStrike.h"
#include "GlobalSettings.h"
#include "GuitarString.h"

#include <cstdio>
using namespace std;

ScoreParserGuitar::ScoreParserGuitar(string filename) : ScoreParser(filename)
{
}

ScoreParserGuitar::~ScoreParserGuitar()
{
}

double ScoreParserGuitar::getDuration()
{
    durationOnly = true;
    duration = -1.0;
    parseInternal();
    return duration;
}

bool ScoreParserGuitar::parse(Instrument *instrument)
{
    durationOnly = false;
    this->instrument = instrument;
    return parseInternal();
}

bool ScoreParserGuitar::parseInternal()
{
    int i, j, k;

    bool foundsomething = false;

    logMessage(1, "ScoreParserGuitar:: attempting to parse %s", filename.c_str());

    FILE *f;

    MatlabParser matlabParser(filename);
    matlabParser.addArray("exc", 5, 0);
    if (!matlabParser.parse()) {
	return false;
    }

    // now look for the required values
    vector<MatlabScalar> *scalars = matlabParser.getScalars();
    vector<MatlabArray> *arrays = matlabParser.getArrays();
    vector<MatlabCellArray> *cellarrays = matlabParser.getCellArrays();

    // scalar Tf is duration
    for (i = 0; i < scalars->size(); i++) {
	if (scalars->at(i).name == "Tf") {
	    duration = scalars->at(i).value;
	}
	else if (scalars->at(i).name == "highpass") {
	    if (scalars->at(i).value != 0) {
		logMessage(1, "Enabling high pass filter\n");
		GlobalSettings::getInstance()->setHighPassOn(true);
	    }
	}
    }
    if (durationOnly) {
	if (duration < 0.0) return false;
	return true;
    }

    // start writing the raw score output
    f = fopen("rawscore.m", "w");
    fprintf(f, "%% Raw score file, automatically generated from %s\n", filename.c_str());
    fprintf(f, "Tf = %f\n\n", duration);

    // find the excitation array
    for (i = 0; i < arrays->size(); i++) {
	if (arrays->at(i).name == "exc") {
	    MatlabArray exc = arrays->at(i);

	    if ((exc.width < 5) || (exc.width > 6)) {
		logMessage(3, "ScoreParserGuitar: exc array is incorrect width");
		fclose(f);
		return false;
	    }

	    if (exc.height > 0) foundsomething = true;

	    fprintf(f, "exc = [ ");

	    for (j = 0; j < exc.height; j++) {
		if (exc.width == 5) {
		    // original version that supports plucks only
		    int stridx = (int)exc.data[(j*5)+0];
		    char strname[20];
		    sprintf(strname, "string%d", stridx);
		    Component *comp = instrument->getComponentByName(strname);
		    if (!comp) {
			logMessage(3, "ScoreParserGuitar: cannot find string %d", stridx);
			fclose(f);
			return false;
		    }
		    InputPluck *pluck = new InputPluck(comp, exc.data[(j*5)+2], 0.0, 0.0,
						       exc.data[(j*5)+1], exc.data[(j*5)+3],
						       exc.data[(j*5)+4], 1, 0);
		    comp->addInput(pluck);
		    
		    fprintf(f, "%.15f  %.15f  %.15f  %.15f  %.15f", exc.data[(j*5)+0], exc.data[(j*5)+1],
			    exc.data[(j*5)+2], exc.data[(j*5)+3], exc.data[(j*5)+4]);
		}
		else {
		    // new version for net1 that supports strikes and plucks
		    int stridx = (int)exc.data[(j*6)+0];
		    char strname[20];
		    sprintf(strname, "string%d", stridx);
		    Component *comp = instrument->getComponentByName(strname);
		    if (!comp) {
			logMessage(3, "ScoreParserGuitar: cannot find string %d", stridx);
			fclose(f);
			return false;
		    }
		    Input *input;
		    if (exc.data[(j*6)+5] == 0.0) {
			// strike
			input = new InputStrike(comp, exc.data[(j*6)+2], 0.0, 0.0,
						exc.data[(j*6)+1], exc.data[(j*6)+3],
						exc.data[(j*6)+4], 1, 0);
		    }
		    else {
			// pluck
			input = new InputPluck(comp, exc.data[(j*6)+2], 0.0, 0.0,
					       exc.data[(j*6)+1], exc.data[(j*6)+3],
					       exc.data[(j*6)+4], 1, 0);
		    }
		    comp->addInput(input);
		    
		    fprintf(f, "%.15f  %.15f  %.15f  %.15f  %.15f  %.15f", exc.data[(j*6)+0], exc.data[(j*6)+1],
			    exc.data[(j*6)+2], exc.data[(j*6)+3], exc.data[(j*6)+4], exc.data[(j*6)+5]);
		}
		if (j < (exc.height - 1)) {
		    fprintf(f, ";\n        ");
		}
		else {
		    fprintf(f, "];\n\n");
		}
	    }
	}
    }

    // now find the finger_def cell array
    for (i = 0; i < cellarrays->size(); i++) {
	if (cellarrays->at(i).name == "finger_def") {
	    foundsomething = true;
	    MatlabCellArray finger_def = cellarrays->at(i);

	    if (finger_def.width != 3) {
		logMessage(3, "ScoreParserGuitar: finger_def is incorrect width");
		fclose(f);
		return false;
	    }

	    fprintf(f, "finger_def = {\n");

	    for (j = 0; j < finger_def.height; j++) {
		// find the correct string
		if (finger_def.data[(j*3)+0].type != CELL_SCALAR) {
		    logMessage(3, "ScoreParserGuitar: expected string index in first column of finger_def");
		    fclose(f);
		    return false;
		}
		int stridx = (int)finger_def.data[(j*3)+0].scalar.value;
		char strname[20];
		sprintf(strname, "string%d", stridx);
		GuitarString *str = dynamic_cast<GuitarString*>(instrument->getComponentByName(strname));
		if (!str) {
		    logMessage(3, "ScoreParserGuitar: cannot find string %d", stridx);
		    fclose(f);
		    return false;
		}

		fprintf(f, "    %d, [", stridx);

		// get the times, positions and forces
		vector<double> times, position, force;
		MatlabCellContent mcc = finger_def.data[(j*3)+1];
		if ((mcc.type != CELL_ARRAY) || (mcc.array.width != 3)) {
		    logMessage(3, "ScoreParserGuitar: expected array of width 3 in second column of finger_def");
		    fclose(f);
		    return false;
		}
		for (k = 0; k < mcc.array.height; k++) {
		    times.push_back(mcc.array.data[(k*3)+0]);
		    position.push_back(mcc.array.data[(k*3)+1]);
		    force.push_back(mcc.array.data[(k*3)+2]);

		    fprintf(f, "%.15f %.15f %.15f", mcc.array.data[(k*3)+0], mcc.array.data[(k*3)+1],
			    mcc.array.data[(k*3)+2]);
		    if (k < (mcc.array.height - 1)) fprintf(f, "; ");
		}

		// get the initial position and velocity
		mcc = finger_def.data[(j*3)+2];
		if ((mcc.type != CELL_ARRAY) || (mcc.array.width != 2) || (mcc.array.height != 1)) {
		    logMessage(3, "ScoreParserGuitar: expected 2x1 array in third column of finger_def");
		    fclose(f);
		    return false;
		}
		double uf0 = mcc.array.data[0];
		double vf0 = mcc.array.data[1];

		// finally add the finger!
		str->addFinger(uf0, vf0, &times, &position, &force);

		fprintf(f, "], [%.15f, %.15f]", uf0, vf0);
		if (j < (finger_def.height - 1)) {
		    fprintf(f, ";\n");
		}
		else {
		    fprintf(f, "\n};\n\n");
		}
	    }
	}
    }

    fclose(f);
    return foundsomething;
}
