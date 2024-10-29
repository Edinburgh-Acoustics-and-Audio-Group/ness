/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "InstrumentParserGuitar.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "Output.h"
#include "GuitarString.h"
#include "MatlabParser.h"
#include "ConnectionNet1.h"

#include <cstdio>
using namespace std;

int InstrumentParserGuitar::string_num = 0;

InstrumentParserGuitar::InstrumentParserGuitar(string filename) : InstrumentParser(filename)
{
}

InstrumentParserGuitar::~InstrumentParserGuitar()
{
}

#define SCALAR_VALUE(namestring, var) \
    if (scalars->at(i).name == namestring) { \
        var = scalars->at(i).value;	       \
    } \


#define ARRAY_VALUE(namestring, var) \
    if (arr.name == namestring) { \
        for (j = 0; j < (arr.width * arr.height); j++) { \
	    var.push_back(arr.data[j]); \
	} \
    } \

Instrument *InstrumentParserGuitar::parse()
{
    int i, j;

    logMessage(1, "InstrumentParserGuitar: attempting to parse %s", filename.c_str());

    instrument = new Instrument();

    // parse the Matlab file
    MatlabParser matlabParser(filename);
    if (!matlabParser.parse()) {
	delete instrument;
	instrument = NULL;
	return NULL;
    }

    // now look for the required values
    vector<MatlabScalar> *scalars = matlabParser.getScalars();
    vector<MatlabArray> *arrays = matlabParser.getArrays();

    double SR = 44100.0;
    vector<double> string_def, output_def, pan, backboard, frets, barrier_params_def, finger_params, ssconnect_def;
    int fretnum = -1;
    int normalize_outs = 0;
    int string_def_width = -1;
    int itnum = 20;

    for (i = 0; i < scalars->size(); i++) {
	SCALAR_VALUE("SR", SR);
	SCALAR_VALUE("fretnum", fretnum);
	SCALAR_VALUE("normalize_outs", normalize_outs);
	SCALAR_VALUE("normalise_outs", normalize_outs);
	SCALAR_VALUE("itnum", itnum);

	// string_num and out_num are defined in the instrument file as well, but
	// Matlab ignores them, so ignore them here as well for consistency
    }
    for (i = 0; i < arrays->size(); i++) {
	MatlabArray arr = arrays->at(i);
	if (arr.name == "string_def") {
	    string_def_width = arr.width;
	}
	ARRAY_VALUE("string_def", string_def);
	ARRAY_VALUE("output_def", output_def);
	ARRAY_VALUE("pan", pan);
	ARRAY_VALUE("backboard", backboard);
	ARRAY_VALUE("frets", frets);
	ARRAY_VALUE("barrier_params_def", barrier_params_def);
	ARRAY_VALUE("finger_params", finger_params);
	ARRAY_VALUE("ssconnect_def", ssconnect_def);
    }

    GlobalSettings::getInstance()->setSampleRate(SR);

    if (normalize_outs) GlobalSettings::getInstance()->setNormaliseOuts(true);

    if (fretnum < 0) fretnum = frets.size() / 2;

    // sanity checking
    if (string_def.size() == 0) {
	logMessage(3, "Error parsing %s as guitar instrument: no strings defined", filename.c_str());
	delete instrument;
	return NULL;
    }
    if ((string_def_width != 6) && (string_def_width != 7)) {
	logMessage(3, "Error parsing %s as guitar instrument: incomplete string_def", filename.c_str());
	delete instrument;
	return NULL;
    }
    if (output_def.size() != (pan.size() * 2)) {
	logMessage(3, "Error parsing %s as guitar instrument: number of outputs doesn't match number of pans",
		   filename.c_str());
	delete instrument;
	return NULL;
    }
    if ((frets.size() % 2) != 0) {
	logMessage(3, "Error parsing %s as guitar instrument: incomplete frets array", filename.c_str());
	delete instrument;
	return NULL;
    }
    if ((backboard.size() != 0) && (backboard.size() != 3)) {
	logMessage(3, "Error parsing %s as guitar instrument: wrong number of backboard parameters",
		   filename.c_str());
	delete instrument;
	return NULL;
    }
    if ((barrier_params_def.size() != 0) && (barrier_params_def.size() != 5)) {
	logMessage(3, "Error parsing %s as guitar instrument: wrong number of barrier parameters",
		   filename.c_str());
	delete instrument;
	return NULL;
    }
    if ((finger_params.size() != 0) && (finger_params.size() != 4)) {
	logMessage(3, "Error parsing %s as guitar instrument: wrong number of finger parameters",
		   filename.c_str());
	delete instrument;
	return NULL;
    }
    if ((ssconnect_def.size() % 9) != 0) {
	logMessage(3, "Error parsing %s as guitar instrument: incomplete connection definition array");
	delete instrument;
	return NULL;
    }

    FILE *f = fopen("rawinstrument.m", "w");
    fprintf(f, "%% gtversion 1.0\n");
    fprintf(f, "%% Raw instrument file, automatically generated from %s\n\n", filename.c_str());
    fprintf(f, "SR = %d;\n\n", (int)SR);

    vector<GuitarString*> strings;

    // create the strings
    fprintf(f, "string_def = [");

    if (string_def_width == 7) {
	// original version with E and rho set directly
	for (i = 0; i < string_def.size(); i += 7) {
	    char namebuf[20];
	    sprintf(namebuf, "string%d", ((i/7)+1));
	    GuitarString *str = new GuitarString(namebuf, string_def[i], string_def[i+1], string_def[i+2],
						 string_def[i+3], string_def[i+4], string_def[i+5],
						 string_def[i+6]);
	    strings.push_back(str);
	    instrument->addComponent(str);
	    
	    fprintf(f, "%.15f %.15f %.15f %.15f %.15f %.15f %.15f", string_def[i], string_def[i+1],
		    string_def[i+2], string_def[i+3], string_def[i+4], string_def[i+5], string_def[i+6]);
	    if (i < (string_def.size() - 7)) {
		fprintf(f, "; ");
	    }
	}
    }
    else {
	// net1 version with materials table
	const double material_tab[9] = {
	    2e11, 0.3, 7850.0,
	    7e10, 0.35, 2700.0,
	    1.6e10, 0.44, 11340.0
	};
	for (i = 0; i < string_def.size(); i += 6) {
	    char namebuf[20];
	    sprintf(namebuf, "string%d", ((i/6)+1));

	    int mat = (int)string_def[i+1] - 1;
	    if ((mat < 0) || (mat > 2)) {
		logMessage(3, "Error parsing %s as guitar instrument: invalid material index in string_def");
		delete instrument;
		fclose(f);
		return NULL;
	    }

	    GuitarString *str = new GuitarString(namebuf, string_def[i], material_tab[mat*3+0], string_def[i+2],
						 string_def[i+3], material_tab[mat*3+2], string_def[i+4],
						 string_def[i+5]);
	    strings.push_back(str);
	    instrument->addComponent(str);
	    
	    fprintf(f, "%.15f %.15f %.15f %.15f %.15f %.15f", string_def[i], string_def[i+1],
		    string_def[i+2], string_def[i+3], string_def[i+4], string_def[i+5]);
	    if (i < (string_def.size() - 6)) {
		fprintf(f, "; ");
	    }
	}
    }
    fprintf(f, "];\nstring_num = %d;\n\n", strings.size());

    string_num = strings.size();

    // create the outputs
    fprintf(f, "output_def = [");
    for (i = 0; i < output_def.size(); i += 2) {
	int stridx = ((int)output_def[i]) - 1;
	if ((stridx < 0) || (stridx >= strings.size())) {
	    logMessage(3, "Error parsing %s as guitar instrument: invalid string index in output_def",
		       filename.c_str());
	    delete instrument;
	    fclose(f);
	    return NULL;
	}
	Output *output = new Output(strings[stridx], pan[i/2], output_def[i+1], 0.0, 0.0, 1);
	instrument->addOutput(output);

	fprintf(f, "%d %f", (int)output_def[i], output_def[i+1]);
	if (i < (output_def.size() - 2)) {
	    fprintf(f, "; ");
	}
    }
    fprintf(f, "];\nout_num = %d;\n", output_def.size() / 2);
    fprintf(f, "normalize_outs = 0;\n\n");

    // write pan information
    fprintf(f, "pan = [");
    for (i = 0; i < pan.size(); i++) {
	fprintf(f, "%.15f  ", pan[i]);
    }
    fprintf(f, "];\n\n");

    // separate frets into two different vectors
    vector<double> fretpos, fretheight;
    for (i = 0; i < frets.size(); i += 2) {
	fretpos.push_back(frets[i]);
	fretheight.push_back(frets[i+1]);
    }

    // setup the strings for collisions
    for (i = 0; i < strings.size(); i++) {
	if (backboard.size() > 0) {
	    strings[i]->setBackboard(backboard[0], backboard[1], backboard[2]);
	}
	if (frets.size() > 0) {
	    strings[i]->setFrets(fretpos, fretheight);
	}
	if (barrier_params_def.size() > 0) {
	    strings[i]->setBarrierParams(barrier_params_def[0], barrier_params_def[1],
					 barrier_params_def[2], (int)barrier_params_def[3],
					 barrier_params_def[4]);
	}
	if (finger_params.size() > 0) {
	    strings[i]->setFingerParams(finger_params[0], finger_params[1], finger_params[2],
					finger_params[3]);
	}
    }

    // write collision information
    if (backboard.size() > 0) {
	fprintf(f, "backboard = [%f %f %f];\n\n", backboard[0], backboard[1], backboard[2]);
    }
    if (frets.size() > 0) {
	fprintf(f, "fretnum = %d;\n", fretpos.size() / 2);
	fprintf(f, "frets = [");
	for (i = 0; i < fretpos.size(); i++) {
	    fprintf(f, "%.15f  %.15f", fretpos[i], fretheight[i]);
	    if (i < (fretpos.size() - 1)) fprintf(f, "; ");
	}
	fprintf(f, "];\n\n");
    }
    if (barrier_params_def.size() > 0) {
	fprintf(f, "barrier_params_def = [%.15f %.15f %.15f %d %.15f];\n", barrier_params_def[0],
		barrier_params_def[1], barrier_params_def[2], (int)barrier_params_def[3],
		barrier_params_def[4]);
	fprintf(f, "itnum = %d;\ntol = %.15f;\n\n", (int)barrier_params_def[3], barrier_params_def[4]);
    }
    if (finger_params.size() > 0) {
	fprintf(f, "finger_params = [%.15f %.15f %.15f %.15f];\n\n", finger_params[0], finger_params[1],
		finger_params[2], finger_params[3]);
    }

    // handle connections
    if (ssconnect_def.size() > 0) {
	vector<ConnectionNet1*> conns;

	fprintf(f, "ssconnect_def = [");
	for (i = 0; i < ssconnect_def.size(); i += 9) {
	    int sstring = (int)ssconnect_def[i+5];
	    int dstring = (int)ssconnect_def[i+7];

	    if ((sstring < 0) || (sstring > strings.size()) ||
		(dstring < 0) || (dstring > strings.size())) {
		logMessage(3, "Error parsing %s as guitar instrument: invalid string index in ssconnect_def",
			   filename.c_str());
		delete instrument;
		fclose(f);
		return NULL;
	    }

	    Component *c1 = NULL;
	    Component *c2 = NULL;
	    if (sstring > 0) c1 = strings[sstring-1];
	    if (dstring > 0) c2 = strings[dstring-1];

	    // create the connection
	    ConnectionNet1 *conn = new ConnectionNet1(c1, c2, ssconnect_def[i],
						      ssconnect_def[i+1],
						      ssconnect_def[i+2],
						      ssconnect_def[i+3],
						      ssconnect_def[i+4],
						      ssconnect_def[i+6],
						      ssconnect_def[i+8]);
	    conn->setIterations(itnum);

	    // don't add co-incident connections
	    for (j = 0; j < conns.size(); j++) {
		if (conns[j]->coincides(conn)) {
		    logMessage(1, "Removing co-incident connection");
		    delete conn;
		    conn = NULL;
		    break;
		}
	    }
	    if (conn) {
		instrument->addConnection(conn);
		conns.push_back(conn);
	    }

	    fprintf(f, "%.15f, %.15f, %.15f, %.15f, %.15f, %d, %.15f, %d, %.15f",
		    ssconnect_def[i], ssconnect_def[i+1], ssconnect_def[i+2],
		    ssconnect_def[i+3], ssconnect_def[i+4], sstring,
		    ssconnect_def[i+6], dstring, ssconnect_def[i+8]);
	    if (i < (ssconnect_def.size()-9)) {
		fprintf(f, "; ");
	    }
	}
	fprintf(f, "];\n\n");
    }

    fclose(f);
    return instrument;
}
