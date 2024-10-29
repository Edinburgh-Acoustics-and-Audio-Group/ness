/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "Profiler.h"
#include "Logger.h"

#include <sstream>
using namespace std;

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

Profiler::Profiler(int size, const char **names)
{
    init(size, names);
}

void Profiler::init(int size, const char **names)
{
    int i;
    this->size = size;
    times = new double[size];
    for (i = 0; i < size; i++) {
	times[i] = 0.0;
    }
    maxUsed = 0;
    this->names = names;
}

Profiler::Profiler()
{
    init(20, NULL);
}

Profiler::~Profiler()
{
    delete[] times;
}

void Profiler::start(int idx)
{
    startTime = getTime();
}

void Profiler::end(int idx)
{
    if (idx < size) {
	times[idx] += getTime() - startTime;
    }
    if (idx > maxUsed) maxUsed = idx;
}

double Profiler::get(int idx)
{
    if (idx >= size) return 0.0;
    return times[idx];
}

double Profiler::getTime()
{
#ifndef WIN32
    const  double  micro = 1.0e-06;    /* Conversion constant */
    double         wall_time;          /* To hold the result */
    
    if ( gettimeofday( &tp, NULL) == -1 )
    	wall_time = -1.0e0;
    else
	wall_time = (double) (tp.tv_sec + micro*tp.tv_usec);
    
    return wall_time;
#else
    return double(GetTickCount64()) / 1000.0f;
#endif
}

string Profiler::print()
{
    int i;
    ostringstream ss;
    for (i = 0; i <= maxUsed; i++) {
	if (!names) {
	    ss << i << ": " << times[i] << endl;
	}
	else {
	    ss << i << ":" << names[i] << ": " << times[i] << endl;
	}
    }
    return ss.str();
}
