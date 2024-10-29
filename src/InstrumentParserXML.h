/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Parser for the native XML instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_XML_H_
#define _INSTRUMENT_PARSER_XML_H_

#include "InstrumentParser.h"

#ifndef BRASS_ONLY
#include "GuitarString.h"
#include "ConnectionNet1.h"
#ifndef GUITAR_ONLY
#include "AirboxIndexed.h"
#include "ComponentString.h"
#include "ConnectionP2P.h"
#include "BowedString.h"
#endif
#endif

#include "libxml/parser.h"
#include "libxml/tree.h"

#include <vector>
using namespace std;

class InstrumentParserXML : public InstrumentParser {
 public:
    InstrumentParserXML(string filename);
    virtual ~InstrumentParserXML();

 protected:
    virtual Instrument *parse();

#ifndef BRASS_ONLY
    bool parseGuitar(xmlNodePtr node);
    bool parseGuitarString(xmlNodePtr node, GuitarString **result);
    bool parseConnectionNet1(xmlNodePtr node);

    vector<ConnectionNet1*> net1Connections;

#ifndef GUITAR_ONLY
    bool parseAirbox(xmlNodePtr node);
    bool parsePlate(xmlNodePtr node);
    bool parseConnection(xmlNodePtr node);
    bool parseDrumShell(xmlNodePtr node, AirboxIndexed *airbox);
    bool parsePlateEmbedded(xmlNodePtr node, Airbox *airbox, bool isMembrane);
    bool parseBar(xmlNodePtr node);
    bool parseComponentString(xmlNodePtr node, ComponentString **result);
    bool parseFretboard(xmlNodePtr node);
    bool parseBowParameters(xmlNodePtr node, double &Kw, double &alpha, double &beta, double &lambda, double &M);
    bool parseFingerParameters(xmlNodePtr node, double &Kw, double &Ku, double &alpha, double &beta, double &lambda, double &M);
    bool parseBowedString(xmlNodePtr node, BowedString **result);
    bool parseModalPlate(xmlNodePtr node);

    vector<ConnectionP2P*> p2pConnections;
#endif
#endif

#ifndef GUITAR_ONLY
    bool parseBrass(xmlNodePtr node);
#endif

    bool parseOutput(xmlNodePtr node);

    bool parseDouble(xmlNodePtr node, double *result);
    bool parseInt(xmlNodePtr node, int *result);
    char *parseString(xmlNodePtr node);
    bool parseBoolean(xmlNodePtr node, bool *result);

    xmlDocPtr doc;
};

#endif
