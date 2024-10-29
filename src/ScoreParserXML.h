/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Score file parser for the native XML format.
 */
#ifndef _SCORE_PARSER_XML_H_
#define _SCORE_PARSER_XML_H_

#include "ScoreParser.h"

#include "libxml/parser.h"
#include "libxml/tree.h"

#include <vector>
using namespace std;

class ScoreParserXML : public ScoreParser {
 public:
    ScoreParserXML(string filename);
    virtual ~ScoreParserXML();
    double getDuration();

 protected:
    virtual bool parse(Instrument *instrument);

    Instrument *instrument;

    char *parseString(xmlNodePtr node);
    bool parseInt(xmlNodePtr node, int *result);
    bool parseDouble(xmlNodePtr node, double *result);

#ifndef BRASS_ONLY
    bool parseStrike(xmlNodePtr node, bool isPluck);
    bool parseFinger(xmlNodePtr node);
#ifndef GUITAR_ONLY
    bool parseBow(xmlNodePtr node);
    bool parseSine(xmlNodePtr node);
#endif
#endif

    bool parseWav(xmlNodePtr node);

#ifndef GUITAR_ONLY
    bool parseLips(xmlNodePtr node);
    bool parseValve(xmlNodePtr node);
#endif

    bool parseBreakpointFunction(xmlNodePtr node, const char *tag1, const char *tag2, vector<double> *result);

    xmlDocPtr doc;
};

#endif
