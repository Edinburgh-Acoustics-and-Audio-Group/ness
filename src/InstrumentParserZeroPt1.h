/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Parser for the Zero point one code instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_ZEROPT1_H_
#define _INSTRUMENT_PARSER_ZEROPT1_H_

#include "InstrumentParser.h"

#include "Plate.h"
#include "Output.h"
#include "ConnectionZeroPt1.h"

class InstrumentParserZeroPt1 : public InstrumentParser {
 public:
    InstrumentParserZeroPt1(string filename);
    virtual ~InstrumentParserZeroPt1();

 protected:
    virtual Instrument *parse();

    virtual int handleItem(string type, istream &in);

    Plate *readPlate(istream &in);
    Output *readOutput(istream &in);
    ConnectionZeroPt1 *readConnection(istream &in);

    vector<ConnectionZeroPt1*> conns;

};

#endif
