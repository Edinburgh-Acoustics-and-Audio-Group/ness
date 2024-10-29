/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Parser for the Matlab brass instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_BRASS_H_
#define _INSTRUMENT_PARSER_BRASS_H_

#include "InstrumentParser.h"

class InstrumentParserBrass : public InstrumentParser {
 public:
    InstrumentParserBrass(string filename);
    virtual ~InstrumentParserBrass();

 protected:
    virtual Instrument *parse();
};

#endif
