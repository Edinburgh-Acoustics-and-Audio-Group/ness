/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "InstrumentParserBrass.h"
#include "BrassInstrument.h"
#include "MatlabParser.h"
#include "Output.h"
#include "Logger.h"
#include "GlobalSettings.h"

InstrumentParserBrass::InstrumentParserBrass(string filename) : InstrumentParser(filename)
{
}

InstrumentParserBrass::~InstrumentParserBrass()
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


Instrument *InstrumentParserBrass::parse()
{
    int i, j;
    logMessage(1, "InstrumentParserBrass: attempting to parse %s", filename.c_str());

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

    vector<double> vpos, vdl, vbl, bore;
    double temperature = 20.0;

    double custominstrument = 0.0;
    double xmeg = -1.0, Leg = -1.0, rmeg = -1.0, rbeg = -1.0, fbeg = -1.0;
    double FS = 44100.0;
    vector<double> x0eg, r0eg;

    for (i = 0; i < scalars->size(); i++) {
	SCALAR_VALUE("custominstrument", custominstrument);
	SCALAR_VALUE("temperature", temperature);
	SCALAR_VALUE("xmeg", xmeg);
	SCALAR_VALUE("Leg", Leg);
	SCALAR_VALUE("rmeg", rmeg);
	SCALAR_VALUE("rbeg", rbeg);
	SCALAR_VALUE("fbeg", fbeg);
	SCALAR_VALUE("FS", FS);
    }

    for (i = 0; i < arrays->size(); i++) {
	MatlabArray arr = arrays->at(i);
	ARRAY_VALUE("vpos", vpos);
	ARRAY_VALUE("vdl", vdl);
	ARRAY_VALUE("vbl", vbl);
	ARRAY_VALUE("bore", bore);

	ARRAY_VALUE("x0eg", x0eg);
	ARRAY_VALUE("r0eg", r0eg);
    }

    GlobalSettings::getInstance()->setSampleRate(FS);

    if ((vpos.size() != vdl.size()) || (vpos.size() != vbl.size())) {
	logMessage(3, "Error parsing brass instrument file: vpos, vdl and vbl must all be same length");
	delete instrument;
	return NULL;
    }

    BrassInstrument *brass;
    if (custominstrument > 0.0) {
	if ((xmeg < 0.0) || (Leg < 0.0) || (rmeg < 0.0) || (rbeg < 0.0) || (fbeg < 0.0) ||
	    (x0eg.size() == 0) || (r0eg.size() == 0)) {
	    logMessage(3, "Error parsing brass instrument file: custom bore parameter missing");
	    delete instrument;
	    return NULL;
	}
	
	brass = new BrassInstrument("brass", temperature, vpos.size(), vpos, vdl, vbl, xmeg,
				    x0eg, Leg, rmeg, r0eg, rbeg, fbeg);
    }
    else {
	if (bore.size() == 0) {
	    logMessage(3, "Error parsing brass instrument file: no bore defined");
	    delete instrument;
	    return NULL;
	}

	brass = new BrassInstrument("brass", temperature, vpos.size(), vpos, vdl, vbl, bore);
    }
    instrument->addComponent(brass);
    instrument->addOutput(new Output(brass, 0.0, 1.0, 0.0, 0.0));

    return instrument;
}
