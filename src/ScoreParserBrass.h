/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Score file parser for the brass format.
 */
#ifndef _SCORE_PARSER_BRASS_H_
#define _SCORE_PARSER_BRASS_H_

#include "ScoreParser.h"

class ScoreParserBrass : public ScoreParser {
 public:
    ScoreParserBrass(string filename);
    virtual ~ScoreParserBrass();
    double getDuration();

 protected:
    virtual bool parse(Instrument *instrument);
    virtual bool parseInternal();

    Instrument *instrument;
    bool durationOnly;
    double duration;

};

#endif
