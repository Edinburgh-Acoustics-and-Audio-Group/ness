/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Score file parser for the Zero code format.
 * Also handles Zero pt 1 as it's so similar
 */
#ifndef _SCORE_PARSER_ZERO_H_
#define _SCORE_PARSER_ZERO_H_

#include "ScoreParser.h"

#include "InputStrike.h"
#include "InputBow.h"
#include "InputWav.h"

#include <iostream>
using namespace std;

class ScoreParserZero : public ScoreParser {
 public:
    ScoreParserZero(string filename);
    virtual ~ScoreParserZero();
    double getDuration();

 protected:
    virtual bool parse(Instrument *instrument);

    Input *parseStrike(istream &in);
    InputBow *parseBow(istream &in);
    InputWav *parseAudio(istream &in);

    Instrument *instrument;
    bool durationOnly;
    double duration;

    virtual int handleItem(string type, istream &in);
};

#endif
