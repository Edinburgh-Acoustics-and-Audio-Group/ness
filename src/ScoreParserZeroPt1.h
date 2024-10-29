/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Score file parser for the Zero point one code format.
 */
#ifndef _SCORE_PARSER_ZEROPT1_H_
#define _SCORE_PARSER_ZEROPT1_H_

#include "ScoreParserZero.h"

class ScoreParserZeroPt1 : public ScoreParserZero {
 public:
    ScoreParserZeroPt1(string filename);
    virtual ~ScoreParserZeroPt1();
};

#endif
