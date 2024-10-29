/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestParser.h"

void TestParser::setUp()
{
    parser = new DummyParser("parsertest.txt");
}

void TestParser::tearDown()
{
    delete parser;
}

void TestParser::testParser()
{
    parser->parse();
}

