/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Score file parser for the MP3D code format.
 */
#ifndef _SCORE_PARSER_MP3D_H_
#define _SCORE_PARSER_MP3D_H_

#include "ScoreParserZero.h"

class ScoreParserMP3D : public ScoreParserZero {
 public:
    ScoreParserMP3D(string filename);
    virtual ~ScoreParserMP3D();
};

#endif
