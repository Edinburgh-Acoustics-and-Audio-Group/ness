/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Parser for the Zero code instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_ZERO_H_
#define _INSTRUMENT_PARSER_ZERO_H_

#include "InstrumentParser.h"
#include "Plate.h"
#include "Output.h"
#include "ConnectionZero.h"

#include <iostream>
#include <vector>
using namespace std;

class InstrumentParserZero : public InstrumentParser {
 public:
    InstrumentParserZero(string filename);
    virtual ~InstrumentParserZero();

 protected:
    virtual Instrument *parse();

    virtual int handleItem(string type, istream &in);

    Plate *readPlate(istream &in);
    Output *readOutput(istream &in);
    ConnectionZero *readConnection(istream &in);

    vector<ConnectionZero*> conns;
};

#endif
