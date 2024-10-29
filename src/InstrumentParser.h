/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Abstract superclass for instrument file parsers.
 */
#ifndef _INSTRUMENT_PARSER_H_
#define _INSTRUMENT_PARSER_H_

#include "Parser.h"
#include "Instrument.h"

#include <string>
using namespace std;

class InstrumentParser : public Parser {
 public:
    InstrumentParser(string filename);
    virtual ~InstrumentParser();

    static Instrument *parseInstrument(string filename, string hint = "");

 protected:
    Instrument *instrument;
    virtual Instrument *parse() = 0;
};

#endif
