/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2015. All rights reserved.
 *
 * Parser for the guitar code instrument file format.
 */
#ifndef _INSTRUMENTPARSERGUITAR_H_
#define _INSTRUMENTPARSERGUITAR_H_

#include "InstrumentParser.h"

class InstrumentParserGuitar : public InstrumentParser {
 public:
    InstrumentParserGuitar(string filename);
    virtual ~InstrumentParserGuitar();

    static int getNumStrings() { return string_num; }

    // for the unit tests
    static void setNumStrings(int ns) { string_num = ns; }

 protected:
    virtual Instrument *parse();
    static int string_num;
};

#endif
