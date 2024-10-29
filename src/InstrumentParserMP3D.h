/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Parser for the MP3D code instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_MP3D_H_
#define _INSTRUMENT_PARSER_MP3D_H_

#include "InstrumentParser.h"

#include "AirboxIndexed.h"

class InstrumentParserMP3D : public InstrumentParser {
 public:
    InstrumentParserMP3D(string filename);
    virtual ~InstrumentParserMP3D();

 protected:
    virtual Instrument *parse();
    virtual int handleItem(string type, istream &in);

    double adjustPos(double x);

    AirboxIndexed *airbox;
};

#endif
