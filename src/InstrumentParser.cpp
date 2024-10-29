/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 */
#include "InstrumentParser.h"

#include "InstrumentParserXML.h"

#ifndef GUITAR_ONLY
#include "InstrumentParserBrass.h"
#endif

#ifndef BRASS_ONLY
#include "InstrumentParserGuitar.h"
#ifndef GUITAR_ONLY
#include "InstrumentParserZero.h"
#include "InstrumentParserZeroPt1.h"
#include "InstrumentParserMP3D.h"
#include "InstrumentParserSoundboard.h"
#include "InstrumentParserBowedString.h"
#include "InstrumentParserModal.h"
#endif
#endif

#include "Logger.h"

InstrumentParser::InstrumentParser(string filename) : Parser(filename)
{
}

InstrumentParser::~InstrumentParser()
{
}

Instrument *InstrumentParser::parseInstrument(string filename, string hint)
{
    Instrument *result = NULL;
    InstrumentParser *parser;

#ifdef BRASS_ONLY
    if (hint == "brass") {
	parser = new InstrumentParserBrass(filename);
	result = parser->parse();
	delete parser;
	return result;
    }

    // no hint, try each parser in turn until we find one that works
    logMessage(1, "Instrument parser: looking for a parser for %s", filename.c_str());

    parser = new InstrumentParserXML(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserBrass(filename);
    result = parser->parse();
    delete parser;
    return result;
#endif

#ifdef GUITAR_ONLY
    if (hint == "guitar") {
	parser = new InstrumentParserGuitar(filename);
	result = parser->parse();
	delete parser;
	return result;
    }

    // no hint, try each parser in turn until we find one that works
    logMessage(1, "Instrument parser: looking for a parser for %s", filename.c_str());

    parser = new InstrumentParserXML(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserGuitar(filename);
    result = parser->parse();
    delete parser;
    return result;
#endif

#ifndef BRASS_ONLY
#ifndef GUITAR_ONLY
    if (hint == "zero") {
	parser = new InstrumentParserZero(filename);
	result = parser->parse();
	delete parser;
	return result;
    }
    else if (hint == "zeroPt1") {
	parser = new InstrumentParserZeroPt1(filename);
	result = parser->parse();
	delete parser;
	return result;	
    }
    else if (hint == "soundboard") {
	parser = new InstrumentParserSoundboard(filename);
	result = parser->parse();
	delete parser;
	return result;
    }
    else if (hint == "mp3d") {
	parser = new InstrumentParserMP3D(filename);
	result = parser->parse();
	delete parser;
	return result;
    }
    else if (hint == "guitar") {
	parser = new InstrumentParserGuitar(filename);
	result = parser->parse();
	delete parser;
	return result;
    }
    else if (hint == "bowedstring") {
	parser = new InstrumentParserBowedString(filename);
	result = parser->parse();
	delete parser;
	return result;
    }
    else if (hint == "modal") {
	parser = new InstrumentParserModal(filename);
	result = parser->parse();
	delete parser;
	return result;	
    }
    else if (hint == "brass") {
	parser = new InstrumentParserBrass(filename);
	result = parser->parse();
	delete parser;
	return result;
    }

    // no hint, try each parser in turn until we find one that works
    logMessage(1, "Instrument parser: looking for a parser for %s", filename.c_str());

    parser = new InstrumentParserXML(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserBowedString(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserZero(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserZeroPt1(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserMP3D(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserGuitar(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserSoundboard(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserModal(filename);
    result = parser->parse();
    delete parser;
    if (result) return result;

    parser = new InstrumentParserBrass(filename);
    result = parser->parse();
    delete parser;

    return result;
#endif
#endif
}
