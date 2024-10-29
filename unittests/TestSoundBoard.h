/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for SoundBoard
 */
#ifndef _TESTSOUNDBOARD_H_
#define _TESTSOUNDBOARD_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "SoundBoard.h"
#include "ComponentString.h"

#include <vector>
using namespace std;

class TestSoundBoard : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testSoundBoard();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("SoundBoardTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestSoundBoard>("testSoundBoard", &TestSoundBoard::testSoundBoard));
        return suiteOfTests;
    }

 private:
    SoundBoard *soundBoard;
    vector<ComponentString*> *strings;

};

#endif
