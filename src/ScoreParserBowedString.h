/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Score file parser for the bowed string format.
 */
#ifndef _SCORE_PARSER_BOWED_STRING_H_
#define _SCORE_PARSER_BOWED_STRING_H_

#include "ScoreParser.h"

class ScoreParserBowedString : public ScoreParser {
 public:
    ScoreParserBowedString(string filename);
    virtual ~ScoreParserBowedString();
    double getDuration();

 protected:
    virtual bool parse(Instrument *instrument);
    virtual bool parseInternal();

    Instrument *instrument;
    bool durationOnly;
    double duration;

};


#endif
