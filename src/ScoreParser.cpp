/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "ScoreParser.h"
#include "ScoreParserXML.h"

#ifndef GUITAR_ONLY
#include "ScoreParserBrass.h"
#endif

#ifndef BRASS_ONLY
#include "ScoreParserGuitar.h"
#ifndef GUITAR_ONLY
#include "ScoreParserZero.h"
#include "ScoreParserZeroPt1.h"
#include "ScoreParserMP3D.h"
#include "ScoreParserSoundboard.h"
#include "ScoreParserBowedString.h"
#include "ScoreParserModal.h"
#endif
#endif

ScoreParser::ScoreParser(string filename) : Parser(filename)
{
}

ScoreParser::~ScoreParser()
{
}

bool ScoreParser::parseScore(string filename, Instrument *instrument, string hint)
{
    bool result = false;
    ScoreParser *parser;

#ifdef BRASS_ONLY
    if (hint == "brass") {
	parser = new ScoreParserBrass(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;	
    }

    // no hint, try each parser in turn until we find one that works
    parser = new ScoreParserXML(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserBrass(filename);
    result = parser->parse(instrument);
    delete parser;
    return result;
#endif

#ifdef GUITAR_ONLY
    if (hint == "guitar") {
	parser = new ScoreParserGuitar(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }

    // no hint, try each parser in turn until we find one that works
    parser = new ScoreParserXML(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserGuitar(filename);
    result = parser->parse(instrument);
    delete parser;
    return result;
#endif

#ifndef BRASS_ONLY
#ifndef GUITAR_ONLY
    if (hint == "zero") {
	parser = new ScoreParserZero(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }
    else if (hint == "zeroPt1") {
	parser = new ScoreParserZeroPt1(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }
    else if (hint == "soundboard") {
	parser = new ScoreParserSoundboard(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }
    else if (hint == "mp3d") {
	parser = new ScoreParserMP3D(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }
    else if (hint == "guitar") {
	parser = new ScoreParserGuitar(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }
    else if (hint == "bowedstring") {
	parser = new ScoreParserBowedString(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }
    else if (hint == "brass") {
	parser = new ScoreParserBrass(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;	
    }
    else if (hint == "modal") {
	parser = new ScoreParserModal(filename);
	result = parser->parse(instrument);
	delete parser;
	return result;
    }

    // no hint, try each parser in turn until we find one that works
    parser = new ScoreParserXML(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserZero(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserZeroPt1(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserMP3D(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserBowedString(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserGuitar(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserSoundboard(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserBrass(filename);
    result = parser->parse(instrument);
    delete parser;
    if (result) return result;

    parser = new ScoreParserModal(filename);
    result = parser->parse(instrument);
    delete parser;

    return result;
#endif
#endif
}

double ScoreParser::getDuration(string filename, string hint)
{
    double duration;
#ifdef BRASS_ONLY
    if (hint == "brass") {
	ScoreParserBrass *parserBrass = new ScoreParserBrass(filename);
	duration = parserBrass->getDuration();
	delete parserBrass;
	return duration;
    }

    // try XML format first...
    ScoreParserXML *parserXML = new ScoreParserXML(filename);
    duration = parserXML->getDuration();
    delete parserXML;
    if (duration > 0.0) return duration;

    ScoreParserBrass *parserBrass = new ScoreParserBrass(filename);
    duration = parserBrass->getDuration();
    delete parserBrass;
    return duration;
#endif

#ifdef GUITAR_ONLY
    if (hint == "guitar") {
	ScoreParserGuitar *parserGuitar = new ScoreParserGuitar(filename);
	duration = parserGuitar->getDuration();
	delete parserGuitar;
	return duration;
    }

    // try XML format first...
    ScoreParserXML *parserXML = new ScoreParserXML(filename);
    duration = parserXML->getDuration();
    delete parserXML;
    if (duration > 0.0) return duration;

    ScoreParserGuitar *parserGuitar = new ScoreParserGuitar(filename);
    duration = parserGuitar->getDuration();
    delete parserGuitar;
    return duration;
#endif

#ifndef BRASS_ONLY
#ifndef GUITAR_ONLY
    if ((hint == "zero") || (hint == "zeroPt1") || (hint == "soundboard") ||
	(hint == "mp3d")) {
	ScoreParserZero *parserZero = new ScoreParserZero(filename);
	duration = parserZero->getDuration();
	delete parserZero;
	return duration;
    }
    else if (hint == "bowedstring") {
	ScoreParserBowedString *parserBowedString = new ScoreParserBowedString(filename);
	duration = parserBowedString->getDuration();
	delete parserBowedString;
	return duration;
    }
    else if (hint == "guitar") {
	ScoreParserGuitar *parserGuitar = new ScoreParserGuitar(filename);
	duration = parserGuitar->getDuration();
	delete parserGuitar;
	return duration;
    }
    else if (hint == "brass") {
	ScoreParserBrass *parserBrass = new ScoreParserBrass(filename);
	duration = parserBrass->getDuration();
	delete parserBrass;
	return duration;
    }
    else if (hint == "modal") {
	ScoreParserModal *parserModal = new ScoreParserModal(filename);
	duration = parserModal->getDuration();
	delete parserModal;
	return duration;
    }

    // try XML format first...
    ScoreParserXML *parserXML = new ScoreParserXML(filename);
    duration = parserXML->getDuration();
    delete parserXML;
    if (duration > 0.0) return duration;

    ScoreParserBrass *parserBrass = new ScoreParserBrass(filename);
    duration = parserBrass->getDuration();
    delete parserBrass;
    if (duration > 0.0) return duration;

    // now try Zero format, which should work for any of the other score file types except guitar
    ScoreParserZero *parserZero = new ScoreParserZero(filename);
    duration = parserZero->getDuration();
    delete parserZero;
    if (duration > 0.0) return duration;

    ScoreParserBowedString *parserBowedString = new ScoreParserBowedString(filename);
    duration = parserBowedString->getDuration();
    delete parserBowedString;
    if (duration > 0.0) return duration;

    ScoreParserGuitar *parserGuitar = new ScoreParserGuitar(filename);
    duration = parserGuitar->getDuration();
    delete parserGuitar;
    if (duration > 0.0) return duration;

    ScoreParserModal *parserModal = new ScoreParserModal(filename);
    duration = parserModal->getDuration();
    delete parserModal;

    return duration;
#endif
#endif
}
