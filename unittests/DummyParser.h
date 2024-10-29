/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Dummy parser subclass for testing
 */

#ifndef _DUMMYPARSER_H_
#define _DUMMYPARSER_H_

#include "Parser.h"

class DummyParser : public Parser {
 public:
    DummyParser(string filename);
    virtual ~DummyParser();

    void parse();

 protected:
    virtual int handleItem(string type, istream &in);

 private:
    int count;
};

#endif
