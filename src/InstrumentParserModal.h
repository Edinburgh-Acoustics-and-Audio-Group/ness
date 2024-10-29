/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Parser for the modal plate code instrument file format.
 */
#ifndef _INSTRUMENT_PARSER_MODAL_H_
#define _INSTRUMENT_PARSER_MODAL_H_

#include "InstrumentParser.h"
#include "ModalPlate.h"
#include "OutputModal.h"

#include <iostream>
using namespace std;

class InstrumentParserModal : public InstrumentParser {
 public:
    InstrumentParserModal(string filename);
    virtual ~InstrumentParserModal();

 protected:
    virtual Instrument *parse();
    virtual int handleItem(string type, istream &in);

    ModalPlate *readModalPlate(istream &in);
    OutputModal *readOutput(istream &in);
};

#endif
