/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */

#include "TestLogger.h"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
using namespace std;

void TestLogger::setUp()
{
    // make sure old file isn't still hanging around
    logClose();
    unlink("testlog.txt");

    // set the log filename and filter level
    putenv("NESS_LOG_LEVEL=3");
    putenv("NESS_LOG_FILE=testlog.txt");
}

void TestLogger::tearDown()
{
}

void TestLogger::testLogger()
{
    // send some stuff to the log file
    logMessage(3, "This should appear in the log");
    logMessage(2, "This should NOT appear");
    logMessage(4, "Testing substitution: %d, %s", 6, "hello!");
    logClose();

    // now read the log file and check its contents
    ifstream in("testlog.txt");
    CPPUNIT_ASSERT(in.good());

    char buf[1000];
    in.getline(buf, 1000);
    CPPUNIT_ASSERT(!strcmp(buf, "This should appear in the log"));
    CPPUNIT_ASSERT(in.good());

    in.getline(buf, 1000);
    CPPUNIT_ASSERT(!strcmp(buf, "Testing substitution: 6, hello!"));
    CPPUNIT_ASSERT(in.good());

    // should now be no more lines in the file
    in.getline(buf, 1000);
    CPPUNIT_ASSERT(!in.good());
    in.close();
}
