/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 *
 * Parses a subset of Matlab files, potentially allowing instruments and
 * scores to be shared between C/CUDA and Matlab codes.
 *
 * Can deal with files containing scalar, array, cell array and structure
 * initialisations. Simple scalar expressions are supported but expressions
 * on other types are not. Some simple functions (e.g. sin, cos, sqrt, exp)
 * are provided.
 */

#ifndef _MATLABPARSER_H_
#define _MATLABPARSER_H_

#include <string>
#include <vector>
using namespace std;

class MatlabFunction;

// token types
enum {
    TOKEN_EOF,
    TOKEN_CRLF,
    
    TOKEN_COMMA,
    TOKEN_EQUALS,
    TOKEN_SEMICOLON,
    TOKEN_LBRACKET,
    TOKEN_RBRACKET,
    TOKEN_ELLIPSIS,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_SINGLEQUOTE,
    TOKEN_DOT,

    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_POWER,

    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_STRING
};

struct MatlabToken {
    int type;
    string stringVal;
    double numVal;
    int whitespace;
};

struct MatlabScalar {
    string name;
    double value;
};

struct MatlabArray {
    string name;
    int width;
    int height;
    double *data;
};

struct MatlabString {
    string name;
    string value;
};

enum {
    CELL_SCALAR, CELL_ARRAY, CELL_STRING
};

struct MatlabCellContent {
    int type;              // see enum above
    MatlabScalar scalar;   // only valid if CELL_SCALAR
    MatlabArray array;     // only valid if CELL_ARRAY
    MatlabString string;   // only valid if CELL_STRING
};

struct MatlabCellArray {
    string name;
    int width;
    int height;
    MatlabCellContent *data;
};

// an individual structure (not array)
struct MatlabStructElement {
    vector<string> memberNames;
    vector<MatlabCellContent> memberValues;
};

// top level struct is an array of individual structures
struct MatlabStruct {
    string name;
    vector<MatlabStructElement> elements;
};

class MatlabParser {
 public:
    MatlabParser(string filename);
    virtual ~MatlabParser();

    bool parse();

    void addScalar(string name, double value);
    void addArray(string name, int width, int height, double *data = NULL);
    void addStruct(string name);

    vector<MatlabScalar> *getScalars() { return &scalars; }
    vector<MatlabArray> *getArrays() { return &arrays; }
    vector<MatlabCellArray> *getCellArrays() { return &cellarrays; }
    vector<MatlabStruct> *getStructs() { return &structs; }
    vector<MatlabString> *getStrings() { return &strings; }

 protected:
    bool getNextToken(MatlabToken *tok);
    bool peekNextToken(MatlabToken *tok);

    bool parseValue(MatlabCellContent *val, bool *existingArray = NULL);

    bool parseExpression(MatlabCellContent *val);

    bool parseIndexList(vector<int> &indices, string idname, int max = 2, int end = TOKEN_RPAREN);

    bool handleArrayReference(MatlabArray *arr, vector<int> &indices, MatlabCellContent *val);
    bool handleStructReference(MatlabStruct *str, int idx, string membername, MatlabCellContent *val);

    bool getVariable(string name, MatlabCellContent *val);

    void deleteVariable(string name);

    bool isExpressionStart(MatlabToken &tok);

    MatlabFunction *getFunction(string name);

    MatlabArray *getArray(string name);
    MatlabStruct *getStruct(string name);
    MatlabCellArray *getCellArray(string name);

    void deepCopyArray(MatlabArray &dest, MatlabArray &src);

    char *d;
    int len;
    int pos;

    vector<MatlabScalar> scalars;
    vector<MatlabArray> arrays;
    vector<MatlabCellArray> cellarrays;
    vector<MatlabStruct> structs;
    vector<MatlabString> strings;

    bool inSpaceSeparatedArray;
};

#endif
