/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Parser for the Matlab bowed string instrument file format
 */
#ifndef _INSTRUMENT_PARSER_BOWED_STRING_H_
#define _INSTRUMENT_PARSER_BOWED_STRING_H_

#include "InstrumentParser.h"

class InstrumentParserBowedString : public InstrumentParser {
 public:
    InstrumentParserBowedString(string filename);
    virtual ~InstrumentParserBowedString();

 protected:
    virtual Instrument *parse();
};

#endif

