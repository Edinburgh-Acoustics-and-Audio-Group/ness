/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */
#include "Logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
using namespace std;

static bool logInitialised = false;

// only messages above or equal to this level will be printed
static int logLevel = 5;

// file to log to
static FILE *logFile = NULL;

void logMessage(int level, const char *msg, ...)
{
    va_list ap;
    char *buffer;
    int size;
    int sizeneeded;

    va_start(ap, msg);

    if (!logInitialised) {
	// read log level and log filename from environment
	char *lev = getenv("NESS_LOG_LEVEL");
	if (lev) {
	    logLevel = atoi(lev);
	}
	char *fn = getenv("NESS_LOG_FILE");
	if (fn) {
	    logFile = fopen(fn, "w");	    
	    if (!logFile) {
		fprintf(stderr, "Cannot open log file %s, using stderr instead\n", fn);
		logFile = stderr;
	    }
	}
	else {
	    // defaults to stderr if no file specified
	    logFile = stderr;
	}
	logInitialised = true;
    }

    if (level >= logLevel)
    {
	size = 1024;
	buffer = new char[size];
	if (!buffer)
	{
	    fprintf(stderr, "Out of memory in logMessage");
	    exit(1);
	}
	
	sizeneeded = vsnprintf(buffer, size, msg, ap);
	while (sizeneeded > size)
	{
	    delete[] buffer;
	    size = sizeneeded + 1024;
	    buffer = new char[size];
	    if (!buffer)
	    {
		fprintf(stderr, "Out of memory in logMessage");
		exit(1);
	    }
	    
	    sizeneeded = vsnprintf(buffer, size, msg, ap);
	}

	fprintf(logFile, "%s\n", buffer);
	fflush(logFile);
	delete[] buffer;
    }

    va_end(ap);    
}

void logClose()
{
    if (logFile != stderr) {
	fclose(logFile);
    }
    logInitialised = false;
}


