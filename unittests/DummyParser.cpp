/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "DummyParser.h"

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

DummyParser::DummyParser(string filename)
    : Parser(filename)
{
    count = 0;
}

DummyParser::~DummyParser()
{
}

void DummyParser::parse()
{
    parseTextFile();
}

int DummyParser::handleItem(string type, istream &in)
{
    string stringval;
    int intval;
    double doubleval;

    switch (count) {
    case 0:
	CPPUNIT_ASSERT("token" == type);
	in >> stringval >> intval;
	CPPUNIT_ASSERT(!in.fail());
	CPPUNIT_ASSERT(stringval == "arg1");
	CPPUNIT_ASSERT_EQUAL(2, intval);
	in >> stringval;
	CPPUNIT_ASSERT(in.eof());
	break;
    case 1:
	CPPUNIT_ASSERT("anothertoken" == type);
	in >> stringval >> doubleval;
	CPPUNIT_ASSERT(!in.fail());
	CPPUNIT_ASSERT(stringval == "arg3");
	CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, doubleval, 1e-12);
	in >> stringval;
	CPPUNIT_ASSERT(in.eof());
	break;
    case 2:
	CPPUNIT_ASSERT("finaltoken" == type);
	in >> stringval;
	CPPUNIT_ASSERT(in.eof());
	break;
    default:
	// should only be called 3 times
	CPPUNIT_ASSERT(false);
	break;
    }
    count++;
    return 1;
}
