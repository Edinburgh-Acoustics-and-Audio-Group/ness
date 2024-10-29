/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Parser for the soundboard code instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_SOUNDBOARD_H_
#define _INSTRUMENT_PARSER_SOUNDBOARD_H_

#include "InstrumentParser.h"
#include "StringWithFrets.h"
#include "SoundBoard.h"
#include "Output.h"

class InstrumentParserSoundboard : public InstrumentParser {
 public:
    InstrumentParserSoundboard(string filename);
    virtual ~InstrumentParserSoundboard();

 protected:
    virtual Instrument *parse();

    virtual int handleItem(string type, istream &in);

    StringWithFrets *readString(istream &in);
    SoundBoard *readSoundBoard(istream &in);
    Output *readStringOutput(istream &in);
    Output *readBoardOutput(istream &in);

    vector<StringWithFrets*> strings;
};

#endif
