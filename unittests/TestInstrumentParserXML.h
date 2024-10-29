/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for InstrumentParserXML
 */
#ifndef _TESTINSTRUMENTPARSERXML_H_
#define _TESTINSTRUMENTPARSERXML_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "InstrumentParserXML.h"

class TestInstrumentParserXML : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testInstrumentParserXML();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("InstrumentParserXMLTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestInstrumentParserXML>("testInstrumentParserXML", &TestInstrumentParserXML::testInstrumentParserXML));
        return suiteOfTests;
    }

};

#endif
