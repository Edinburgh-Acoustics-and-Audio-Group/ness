/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Abstract superclass for score file parsers.
 */
#ifndef _SCORE_PARSER_H_
#define _SCORE_PARSER_H_

#include "Parser.h"
#include "Instrument.h"

#include <string>
using namespace std;

class ScoreParser : public Parser {
 public:
    virtual ~ScoreParser();

    static bool parseScore(string filename, Instrument *instrument, string hint = "");

    static double getDuration(string filename, string hint = "");

 protected:
    ScoreParser(string filename);

    virtual bool parse(Instrument *instrument) = 0;
};

#endif
