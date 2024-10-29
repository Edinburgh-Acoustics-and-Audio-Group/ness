/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Score file parser for the guitar format.
 */
#ifndef _SCORE_PARSER_GUITAR_H_
#define _SCORE_PARSER_GUITAR_H_

#include "ScoreParser.h"

class ScoreParserGuitar : public ScoreParser {
 public:
    ScoreParserGuitar(string filename);
    virtual ~ScoreParserGuitar();
    double getDuration();

 protected:
    virtual bool parse(Instrument *instrument);
    virtual bool parseInternal();

    Instrument *instrument;
    bool durationOnly;
    double duration;

};


#endif
