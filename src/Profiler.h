/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Profiler object
 */

#ifndef _PROFILER_H_
#define _PROFILER_H_

#ifndef WIN32
#include <sys/time.h>
#endif

#include <string>
using namespace std;

class Profiler {
 public:
    Profiler(int size, const char **names = NULL);
    Profiler();
    virtual ~Profiler();

    void start(int idx);
    void end(int idx);
    double get(int idx);
    string print();

    double getTime();

 protected:
    void init(int size, const char **names);

    const char **names;

    double *times;
    double startTime;
    int size;
    int maxUsed;

#ifndef WIN32
    struct timeval tp;
#endif
};

#endif
