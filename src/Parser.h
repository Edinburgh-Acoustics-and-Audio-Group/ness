/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Parser superclass, providing text file parsing abilities to the instrument and score
 * parsers.
 */

#ifndef _PARSER_H_
#define _PARSER_H_

#include <string>
#include <iostream>
using namespace std;

class Parser {
 public:
    Parser(string filename);
    virtual ~Parser();

 protected:
    string filename;

    int parseTextFile();

    virtual int handleItem(string type, istream &in);
};

#endif
