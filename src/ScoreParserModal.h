/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Score file parser for the modal plate code.
 */
#ifndef _SCOREPARSERMODAL_H_
#define _SCOREPARSERMODAL_H_

#include "ScoreParser.h"

#include <iostream>
using namespace std;

class ScoreParserModal : public ScoreParser {
 public:
    ScoreParserModal(string filename);
    virtual ~ScoreParserModal();
    double getDuration();

 protected:
    virtual bool parse(Instrument *instrument);

    bool parseStrike(istream &in);
    bool parseSine(istream &in);

    Instrument *instrument;
    bool durationOnly;

    double tail, bpm, currtime;
    double lastStrikeLength;

    virtual int handleItem(string type, istream &in);
};

#endif
