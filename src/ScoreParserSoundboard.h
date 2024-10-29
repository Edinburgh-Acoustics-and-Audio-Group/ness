/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Score file parser for the soundboard code format.
 */
#ifndef _SCORE_PARSER_SOUNDBOARD_H_
#define _SCORE_PARSER_SOUNDBOARD_H_

#include "ScoreParser.h"

class ScoreParserSoundboard : public ScoreParser {
 public:
    ScoreParserSoundboard(string filename);
    virtual ~ScoreParserSoundboard();

 protected:
    Instrument *instrument;

    virtual bool parse(Instrument *instrument);

    virtual int handleItem(string type, istream &in);

    
};

#endif
