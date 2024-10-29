/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "MatlabParser.h"
#include "Logger.h"

#include "MatlabFunction.h"

#ifndef BRASS_ONLY
#include "MatlabFunction_StringDefGen.h"
#include "MatlabFunction_FretDefGen.h"
#include "MatlabFunction_ClusterGen.h"
#include "MatlabFunction_StrumGen.h"
#include "MatlabFunction_PluckGen.h"
#include "MatlabFunction_StrumGenMulti.h"
#endif

#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cmath>
using namespace std;


MatlabParser::MatlabParser(string filename)
{
    d = NULL;
    len = 0;
    pos = 0;

    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) {
	logMessage(5, "Error opening Matlab file %s", filename.c_str());
	return;
    }

    // read entire file contents into d
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);

    d = new char[len+1];
    fread(d, 1, len, f);
    fclose(f);
    d[len] = 0;

    // define pi
    addScalar("pi", 3.14159265358979);

    inSpaceSeparatedArray = false;
}

MatlabParser::~MatlabParser()
{
    int i, j, k;

    for (i = 0; i < arrays.size(); i++) {
	delete[] arrays[i].data;
    }
    for (i = 0; i < cellarrays.size(); i++) {
	for (j = 0; j < (cellarrays[i].width*cellarrays[i].height); j++) {
	    MatlabCellContent *mcc = &cellarrays[i].data[j];
	    if (mcc->type == CELL_ARRAY) delete[] mcc->array.data;
	}
	delete[] cellarrays[i].data;
    }
    for (i = 0; i < structs.size(); i++) {
	for (j = 0; j < structs[i].elements.size(); j++) {
	    for (k = 0; k < structs[i].elements[j].memberValues.size(); k++) {
		if (structs[i].elements[j].memberValues[k].type == CELL_ARRAY) {
		    delete[] structs[i].elements[j].memberValues[k].array.data;
		}
	    }
	}
    }

    if (d) delete[] d;
}

bool MatlabParser::peekNextToken(MatlabToken *tok)
{
    int oldpos = pos;
    bool result = getNextToken(tok);
    pos = oldpos;
    return result;
}

bool MatlabParser::getNextToken(MatlabToken *tok)
{
    int startidx;

    // set to a value > 0 if the token was preceeded by whitespace (and/or comments)
    tok->whitespace = 0;

    if (!d) return false;

    // skip leading spaces and tabs
    while ((d[pos] == ' ') || (d[pos] == '\t')) {
	tok->whitespace++;
	pos++;
    }

    // skip from comment char to end of line (or file)
    while (d[pos] == '%') {
	tok->whitespace++;
	// skip til we get to end of line/file
	while ((d[pos]) && (d[pos] != 10) && (d[pos] != 13)) pos++;
	// skip until we find something interesting
	while ((d[pos] == 10) || (d[pos] == 13) || (d[pos] == ' ') || (d[pos] == '\t')) pos++;
    }

    // reached end of file
    if (!d[pos]) {
	tok->type = TOKEN_EOF;
	return true;
    }

    if (isdigit(d[pos])) { //((isdigit(d[pos])) || (d[pos] == '-')) {
	// found a number
	startidx = pos;
	while ((d[pos]) && ((isdigit(d[pos])) ||
			    (d[pos] == 'e') ||
			    (d[pos] == 'E') ||
			    (d[pos] == '-') ||
			    (d[pos] == '.'))) {
	    if (d[pos] == '-') {
		if ((d[pos-1] != 'e') &&
		    (d[pos-1] != 'E')) break;
	    }
	    pos++;
	}
	string num(&d[startidx], pos - startidx);
	tok->type = TOKEN_NUMBER;
	tok->numVal = atof(num.c_str());
    }
    else if ((isalpha(d[pos])) || (d[pos] == '_')) {
	// found an identifier
	startidx = pos;
	while ((d[pos]) && ((isalnum(d[pos])) || (d[pos] == '_'))) pos++;
	string strval(&d[startidx], pos - startidx);
	tok->type = TOKEN_IDENTIFIER;
	tok->stringVal = strval;
    }
    else if (d[pos] == '\'') {
	// start of string, or transpose operator?
	pos++;
	startidx = pos;
	// look for end of string
	while ((d[pos]) && (d[pos] != 13) && (d[pos] != 10) && (d[pos] != '\'')) pos++;
	if (d[pos] == '\'') {
	    // it's a string
	    string strval(&d[startidx], pos - startidx);
	    tok->type = TOKEN_STRING;
	    tok->stringVal = strval;
	    pos++;
	}
	else {
	    // it's a transpose operator
	    pos = startidx; // rewind
	    tok->type = TOKEN_SINGLEQUOTE;
	}
    }
    else {
	switch (d[pos++]) {
	case ',':
	    tok->type = TOKEN_COMMA;
	    break;
	case ';':
	    tok->type = TOKEN_SEMICOLON;
	    break;
	case '[':
	    tok->type = TOKEN_LBRACKET;
	    break;
	case ']':
	    tok->type = TOKEN_RBRACKET;
	    break;
	case '=':
	    tok->type = TOKEN_EQUALS;
	    break;
	case '{':
	    tok->type = TOKEN_LBRACE;
	    break;
	case '}':
	    tok->type = TOKEN_RBRACE;
	    break;
	case '(':
	    tok->type = TOKEN_LPAREN;
	    break;
	case ')':
	    tok->type = TOKEN_RPAREN;
	    break;
	case '.':
	    if ((d[pos] != '.') || (d[pos+1] != '.')) {
		tok->type = TOKEN_DOT;
		break;
	    }
	    pos += 2;
	    tok->type = TOKEN_ELLIPSIS;
	    break;
	case '+':
	    tok->type = TOKEN_PLUS;
	    break;
	case '-':
	    tok->type = TOKEN_MINUS;
	    break;
	case '*':
	    tok->type = TOKEN_MULTIPLY;
	    break;
	case '/':
	    tok->type = TOKEN_DIVIDE;
	    break;
	case '^':
	    tok->type = TOKEN_POWER;
	    break;
	case 13: // end of line
	case 10:
	    // skip all consecutive whitespace
	    while ((d[pos] == 13) || (d[pos] == 10) || (d[pos] == ' ') || (d[pos] == '\t')) pos++;
	    tok->type = TOKEN_CRLF;
	    break;

	default:
	    // unrecognised token type
	    logMessage(3, "Unrecognised token '%c' in Matlab file" , d[pos-1]);
	    return false;
	}
    }
    
    return true;
}

// gets the contents of the named variable, or returns false if it doesn't exist
bool MatlabParser::getVariable(string name, MatlabCellContent *val)
{
    // first look in scalars
    int i;
    for (i = 0; i < scalars.size(); i++) {
	if (scalars[i].name == name) {
	    val->type = CELL_SCALAR;
	    val->scalar = scalars[i];
	    return true;
	}
    }

    // now look in arrays
    for (i = 0; i < arrays.size(); i++) {
	if (arrays[i].name == name) {
	    val->type = CELL_ARRAY;
	    val->array = arrays[i];
	    return true;
	}
    }

    // FIXME: should maybe do cell arrays here too, but would need changes to MatlabCellContent
    // to accommodate that

    return false;
}

MatlabArray *MatlabParser::getArray(string name)
{
    int i;
    for (i = 0; i < arrays.size(); i++) {
	if (arrays[i].name == name) {
	    return &arrays[i];
	}
    }
    return NULL;
}

MatlabStruct *MatlabParser::getStruct(string name)
{
    int i;
    for (i = 0; i < structs.size(); i++) {
	if (structs[i].name == name) {
	    return &structs[i];
	}
    }
    return NULL;
}

MatlabCellArray *MatlabParser::getCellArray(string name)
{
    int i;
    for (i = 0; i < cellarrays.size(); i++) {
	if (cellarrays[i].name == name) {
	    return &cellarrays[i];
	}
    }
    return NULL;
}

void MatlabParser::deleteVariable(string name)
{
    int i;
    for (i = 0; i < scalars.size(); i++) {
	if (scalars[i].name == name) scalars[i].name = "<deleted>";
    }
    for (i = 0; i < arrays.size(); i++) {
	if (arrays[i].name == name) arrays[i].name = "<deleted>";
    }
    for (i = 0; i < cellarrays.size(); i++) {
	if (cellarrays[i].name == name) cellarrays[i].name = "<deleted>";
    }
    for (i = 0; i < structs.size(); i++) {
	if (structs[i].name == name) structs[i].name = "<deleted>";
    }
    for (i = 0; i < strings.size(); i++) {
	if (strings[i].name == name) strings[i].name = "<deleted>";
    }
}

void MatlabParser::addScalar(string name, double value)
{
    MatlabScalar scal;
    scal.name = name;
    scal.value = value;
    scalars.push_back(scal);
}

void MatlabParser::addArray(string name, int width, int height, double *data)
{
    int i;
    MatlabArray arr;
    arr.name = name;
    arr.width = width;
    arr.height = height;
    arr.data = new double[width * height];
    if (data) {
	for (i = 0; i < (width*height); i++) {
	    arr.data[i] = data[i];
	}
    }
    else {
	for (i = 0; i < (width*height); i++) {
	    arr.data[i] = 0.0;
	}
    }
    arrays.push_back(arr);
}

void MatlabParser::addStruct(string name)
{
    MatlabStruct str;
    str.name = name;
    structs.push_back(str);
}

// returns true if the token could be the start of an expression
bool MatlabParser::isExpressionStart(MatlabToken &tok)
{
    if ((tok.type == TOKEN_NUMBER) ||
	(tok.type == TOKEN_MINUS) ||
	(tok.type == TOKEN_LPAREN) ||
	(tok.type == TOKEN_IDENTIFIER)) {
	return true;
    }
    return false;
}

// factory method for all functions. Returns NULL if name not recognised
MatlabFunction *MatlabParser::getFunction(string name)
{
    if (name == "sin") {
	return new MatlabFunction_Sin();
    }
    else if (name == "cos") {
	return new MatlabFunction_Cos();
    }
    else if (name == "exp") {
	return new MatlabFunction_Exp();
    }
    else if (name == "sqrt") {
	return new MatlabFunction_Sqrt();
    }
#ifndef BRASS_ONLY
    else if (name == "string_def_gen") {
	return new MatlabFunction_StringDefGen();
    }
    else if (name == "fret_def_gen") {
	return new MatlabFunction_FretDefGen();
    }
    else if (name == "cluster_gen") {
	return new MatlabFunction_ClusterGen();
    }
    else if (name == "strum_gen") {
	return new MatlabFunction_StrumGen();
    }
    else if ((name == "pluck_gen") || (name == "event_gen")) {
	return new MatlabFunction_PluckGen();
    }
    else if (name == "strum_gen_multi") {
	return new MatlabFunction_StrumGenMulti();
    }
#endif
    return NULL;
}


bool MatlabParser::handleArrayReference(MatlabArray *arr, vector<int> &indices, MatlabCellContent *val)
{
    // bounds check indices
    if (indices.size() == 1) {
	// 1D
	if ((indices[0] < 1) || (indices[0] > arr->width)) {
	    logMessage(3, "Array index out of bounds in reference to %s", arr->name.c_str());
	    return false;
	}
	val->type = CELL_SCALAR;
	val->scalar.value = arr->data[indices[0]-1];
    }
    else {
	if ((indices[0] < 1) || (indices[0] > arr->height) ||
	    (indices[1] < 1) || (indices[1] > arr->width)) {
	    logMessage(3, "Array index out of bounds in reference to %s", arr->name.c_str());
	    return false;
	} 			    
	val->type = CELL_SCALAR;
	val->scalar.value = arr->data[(indices[0]-1)*arr->width + indices[1]-1];
    }
    return true;
}


// idx is one-based as in Matlab
bool MatlabParser::handleStructReference(MatlabStruct *str, int idx, string membername, MatlabCellContent *val)
{
    int i;
    
    if ((idx < 1) || (idx > str->elements.size())) {
	logMessage(3, "Error in reference to %s: index out of range", str->name.c_str());
	return false;
    }

    // get value from in struct
    MatlabStructElement &mse = str->elements[idx-1];
    for (i = 0; i < mse.memberNames.size(); i++) {
	if (mse.memberNames[i] == membername) {
	    *val = mse.memberValues[i];
	    break;
	}
    }
    if (i >= mse.memberNames.size()) {
	logMessage(3, "Structure %s has no member named %s", str->name.c_str(), membername.c_str());
	return false;
    }
    return true;
}


// parses a comma-separated list of one or two indices and puts them in indices
// the opening bracket/brace should have already been skipped. end specifies the
// type of the closing one
bool MatlabParser::parseIndexList(vector<int> &indices, string idname, int max, int end)
{
    MatlabToken tok;

    // parse out the indices
    if (!peekNextToken(&tok)) return false;
    while (tok.type != end) {
	MatlabCellContent idx;
	if (!parseValue(&idx)) return false;
	if (idx.type != CELL_SCALAR) {
	    logMessage(3, "Parse error in Matlab array reference to %s: index must be scalar value", idname.c_str());
	    return false;
	}
	indices.push_back((int)idx.scalar.value);
	// next should be a comma or the closing bracket
	if (!getNextToken(&tok)) return false;
	if ((tok.type != end) && (tok.type != TOKEN_COMMA)) {
	    logMessage(3, "Syntax error in array reference to %s", idname.c_str());
	    return false;
	}
    }

    // check index count
    if ((indices.size() < 1) || (indices.size() > max)) {
	logMessage(3, "Error in array reference to %s: only 1 or 2 indices supported", idname.c_str());
	return false;
    }
    return true;
}


// parses an expression and returns its value in val
// only scalar expressions properly supported so far
bool MatlabParser::parseExpression(MatlabCellContent *val)
{
    int i;

    MatlabToken tok;
    MatlabCellContent v1, v2;

    vector<double> values;
    vector<int> operators;

    bool unaryminus;
    bool power = false;
    double powerval;
    bool negatepower;

    while (true) {
	if (!getNextToken(&tok)) return false;

	// handle unary minus here
	unaryminus = false;
	while (tok.type == TOKEN_MINUS) {
	    unaryminus = !unaryminus;
	    if (!getNextToken(&tok)) return false;
	}

	if (tok.type == TOKEN_IDENTIFIER) {
	    // a function call or a variable reference
	    string idname = tok.stringVal;

	    if (!peekNextToken(&tok)) return false;
	    if (tok.type == TOKEN_LPAREN) {
		// function call, array reference, or struct reference
		if (!getNextToken(&tok)) return false;

		// check for function first
		MatlabFunction *func = getFunction(idname);
		if (func) {
		    // handle function call
		    vector<MatlabCellContent> params;

		    // parse the parameter list
		    if (!peekNextToken(&tok)) return false;
		    // check for empty list
		    if (tok.type == TOKEN_RPAREN) {
			// consume right parenthesis
			if (!getNextToken(&tok)) return false;
		    }
		    else {
			do {
			    MatlabCellContent par;
			    if (!parseValue(&par)) return false;
			    params.push_back(par);
			    
			    // next should be a comma or the closing parenthesis
			    if (!getNextToken(&tok)) return false;
			    if ((tok.type != TOKEN_RPAREN) && (tok.type != TOKEN_COMMA)) {
				logMessage(3, "Syntax error in function call to %s", idname.c_str());
				return false;
			    }
			} while (tok.type != TOKEN_RPAREN);
		    }

		    // check parameter count
		    // check the parameter count
		    if (params.size() < func->getMinParams()) {
			logMessage(3, "Too few parameters for function %s", idname.c_str());
			delete func;
			return false;
		    }
		    if (params.size() > func->getMaxParams()) {
			logMessage(3, "Too many parameters for function %s", idname.c_str());
			delete func;
			return false;
		    }
		    
		    // set the parameters
		    for (i = 0; i < params.size(); i++) {
			func->setParameter(i, params[i]);
		    }
		    
		    // call the function
		    if (!func->execute(&v1)) {
			delete func;
			return false;
		    }
		    
		    // delete
		    delete func;
		}
		else {
		    // handle array or structure reference
		    vector<int> indices;
       		    if (!parseIndexList(indices, idname)) return false;

		    // find the array/struct
		    MatlabArray *arr = getArray(idname);
		    if (arr) {
			// found array
			if (!handleArrayReference(arr, indices, &v1)) return false;
		    }
		    else {
			// couldn't find array, look for structure
			MatlabStruct *str = getStruct(idname);
			if (!str) {
			    logMessage(3, "Matlab parse error: unrecognised array %s", idname.c_str());
			    return false;
			}

			// parse rest of struct reference
			if (!getNextToken(&tok)) return false;
			if (tok.type != TOKEN_DOT) {
			    logMessage(3, "Parse error in structure reference to %s: expected '.'", idname.c_str());
			    return false;
			}
			if (!getNextToken(&tok)) return false;
			if (tok.type != TOKEN_IDENTIFIER) {
			    logMessage(3, "Parse error in structure reference to %s: expected identifier", idname.c_str());
			    return false;
			}
			string membername = tok.stringVal;

			// bounds check index
			if (indices.size() != 1) {
			    logMessage(3, "Error in Matlab file: structures must be one dimensional");
			    return false;
			}

			if (!handleStructReference(str, indices[0], membername, &v1)) return false;
		    }
		}
	    }
	    else if (tok.type == TOKEN_LBRACE) {
		// cell array reference
		if (!getNextToken(&tok)) return false;

		// find cell array
		MatlabCellArray *carr = getCellArray(idname);
		if (!carr) {
		    logMessage(3, "Matlab parse error: unrecognised cell array name %s", idname.c_str());
		    return false;
		}

		// parse indices
		vector<int> indices;
		if (!parseIndexList(indices, idname, 2, TOKEN_RBRACE)) return false;

		// bounds check indices
		if (indices.size() == 1) {
		    if ((indices[0] < 1) || (indices[0] > carr->width)) {
			logMessage(3, "Cell array index out of bounds in reference to %s", idname.c_str());
			return false;			
		    }
		    v1 = carr->data[indices[0]-1];
		}
		else {
		    if ((indices[0] < 1) || (indices[0] > carr->height) ||
			(indices[1] < 1) || (indices[1] > carr->width)) {
			logMessage(3, "Cell array index out of bounds in reference to %s", idname.c_str());
			return false;
		    }
		    v1 = carr->data[(indices[0]-1)*carr->width + indices[1]-1];
		}
	    }
	    else if (tok.type == TOKEN_DOT) {
		// struct reference
		if (!getNextToken(&tok)) return false;

		// get struct member name
		if (!getNextToken(&tok)) return false;
		if (tok.type != TOKEN_IDENTIFIER) {
		    logMessage(3, "Parse error in Matlab file: expected member name in reference to %s", idname.c_str());
		    return false;
		}
		string membername = tok.stringVal;

		// find the structure
		MatlabStruct *str = getStruct(idname);
		if (!str) {
		    logMessage(3, "Unrecognised structure %s in Matlab file", idname.c_str());
		    return false;
		}

		if (!handleStructReference(str, 1, membername, &v1)) return false;
	    }
	    else {
		// a basic variable reference
		if (!getVariable(idname, &v1)) {
		    logMessage(3, "Unknown variable %s in Matlab file", idname.c_str());
		    return false;
		}
	    }
	}
	else if (tok.type == TOKEN_NUMBER) {
	    // scalar value
	    v1.type = CELL_SCALAR;
	    v1.scalar.value = tok.numVal;
	}
	else if (tok.type == TOKEN_LPAREN) {
	    // left parenthesis
	    // parse the inner expression
	    if (!parseExpression(&v1)) return false;

	    // should be a right parenthesis next
	    if (!getNextToken(&tok)) return false;
	    if (tok.type != TOKEN_RPAREN) {
		logMessage(3, "Syntax error in Matlab parser, expected ')'");
		return false;
	    }
	}
	else {
	    // unrecognised
	    logMessage(3, "Syntax error in Matlab parser, unexpected token in expression");
	    return false;
	}
	// got the value of that term in v1 now
	// handle array element reference
	if (v1.type == CELL_ARRAY) {
	    if (!peekNextToken(&tok)) return false;
	    if (tok.type == TOKEN_LPAREN) {
		vector<int> indices;
		// do array reference
		if (!getNextToken(&tok)) return false; // consume opening bracket
		// parse the indices
		if (!parseIndexList(indices, "")) return false;
		// get the value out
		if (!handleArrayReference(&v1.array, indices, &v2)) return false;
		// copy the value into v1
		v1 = v2;
	    }
	}

	if (v1.type != CELL_SCALAR) {
	    // non-scalars are allowed, but only basic references to them with no computations
	    *val = v1;
	    return true;
	}

	// finish the power operation if one is in progress
	if (power) {
	    power = false;
	    // negate second operand if applicable
	    if (unaryminus) {
		v1.scalar.value = -v1.scalar.value;
	    }
	    // do operation
	    v1.scalar.value = pow(powerval, v1.scalar.value);
	    // set unaryminus to negate result of power
	    unaryminus = negatepower;
	}

	// we parsed a value. next should be an operator, or the end of the expression
	if (!peekNextToken(&tok)) return false;
	if (tok.type == TOKEN_POWER) {
	    // slightly hacky way to handle power operation, because it's higher priority
	    // than the unary minus

	    // skip the power operator
	    if (!getNextToken(&tok)) return false;

	    // flag that we're doing a power operation
	    power = true;

	    // store the info we need
	    powerval = v1.scalar.value;
	    negatepower = unaryminus;

	    // loop back to fetch second power operand
	    continue;
	}

	// negate if there was a unary minus on this value
	if (unaryminus) {
	    v1.scalar.value = -v1.scalar.value;
	}
	values.push_back(v1.scalar.value);

	// hack to make space-separated arrays work
	if ((inSpaceSeparatedArray) && (tok.whitespace > 0)) {
	    break;
	}

	if ((tok.type == TOKEN_PLUS) ||
	    (tok.type == TOKEN_MINUS) ||
	    (tok.type == TOKEN_MULTIPLY) ||
	    (tok.type == TOKEN_DIVIDE)) {
	    // found an operator
	    operators.push_back(tok.type);

	    // consume the token
	    if (!getNextToken(&tok)) return false;
	}
	else {
	    // end of this expression
	    break;
	}
    }

    // now we have a list of operators and a list of values. apply them
    // first pass, do multiplications and divisions
    bool didsomething;
    do {
	didsomething = false;
	for (i = 0; i < operators.size(); i++) {
	    if (operators[i] == TOKEN_MULTIPLY) {
		values[i] = values[i] * values[i+1];
		values.erase(values.begin() + i + 1);
		operators.erase(operators.begin() + i);
		didsomething = true;
		break;
	    }
	    else if (operators[i] == TOKEN_DIVIDE) {
		values[i] = values[i] / values[i+1];
		values.erase(values.begin() + i + 1);
		operators.erase(operators.begin() + i);
		didsomething = true;
		break;
	    }
	}
    } while (didsomething);

    // now do addition and subtraction
    do {
	didsomething = false;
	for (i = 0; i < operators.size(); i++) {
	    if (operators[i] == TOKEN_PLUS) {
		values[i] = values[i] + values[i+1];
		values.erase(values.begin() + i + 1);
		operators.erase(operators.begin() + i);
		didsomething = true;
		break;
	    }
	    else if (operators[i] == TOKEN_MINUS) {
		values[i] = values[i] - values[i+1];
		values.erase(values.begin() + i + 1);
		operators.erase(operators.begin() + i);
		didsomething = true;
		break;
	    }
	}
    } while (didsomething);

    // should now be down to a single value in values
    if (values.size() != 1) {
	logMessage(3, "Internal error in Matlab parser: expected 1 expression value, got %d", values.size());
	return false;
    }

    val->type = CELL_SCALAR;
    val->scalar.value = values[0];

    return true;
}


// parses a scalar or array value
// if the existingArray parameter is not NULL, it receives "true" when a pointer
// to an existing array is returned (indicating that the caller should possibly
// deep-copy it depending on what it wants it for). "false" if a scalar is
// returned, or a new array
bool MatlabParser::parseValue(MatlabCellContent *val, bool *existingArray)
{
    MatlabToken tok, tok2;
    int i;

    if (existingArray) *existingArray = false;

    if (!peekNextToken(&tok)) return false;
    if (isExpressionStart(tok)) {
	if (!parseExpression(val)) return false;

	if ((val->type == CELL_ARRAY) && (existingArray)) *existingArray = true;
    }
    else {
	if (!getNextToken(&tok)) return false;
	if (tok.type == TOKEN_LBRACKET) {
	    //logMessage(1, "Starting array parse");
	    // array value
	    vector<double> arrayvals;
	    int rowlength = -1;

	    val->type = CELL_ARRAY;

	    // check for empty array
	    if (!peekNextToken(&tok)) return false;
	    if (tok.type == TOKEN_RBRACKET) {
		if (!getNextToken(&tok)) return false; // consume the bracket
		val->array.width = 0;
		val->array.height = 0;
		val->array.data = new double[1];
		return true;
	    }

	    /*
	     * Check whether array is comma-separated or not!
	     *
	     * This is all a bit of a hack, but it's one Matlab itself must have as well
	     * so we don't really have any choice.
	     * The problem is, Matlab arrays can be space-separated instead of comma-separated,
	     * and if that's the case, white space can become significant:
	     *
	     *   arr = [-1 -2 -3];  should be parsed as   arr = [-1,-2,-3];
	     *
	     * but
	     *
	     *   arr = [-1-2-3];    should be parsed as   arr = [-6];
	     *
	     * So: when we start an array, we scan through it and set the inSpaceSeparatedArray
	     * flag if there aren't any commas. Then the expression parser can check the flag
	     * and terminate the expression when it encounters whitespace when necessary.
	     * 
	     */
	    inSpaceSeparatedArray = true;
	    int oldpos = pos;
	    if (!getNextToken(&tok)) return false;
	    while (tok.type != TOKEN_RBRACKET) {
		if (tok.type == TOKEN_COMMA) {
		    inSpaceSeparatedArray = false;
		}
		if (!getNextToken(&tok)) return false;
	    }
	    pos = oldpos; // rewind tokeniser

	    while (true) {
		MatlabCellContent v1;

		// parse the next value
		if (!parseExpression(&v1)) return false;
		if (v1.type != CELL_SCALAR) {
		    logMessage(3, "Parse error in Matlab array: expected scalar value");
		    return false;
		}
		arrayvals.push_back(v1.scalar.value);
	    
		// handle arrays with no commas separating the numbers
		if (!peekNextToken(&tok)) return false;
		if (isExpressionStart(tok)) continue;

		// next should be comma, semi-colon, closing bracket
		if (!getNextToken(&tok)) return false;
		if (tok.type == TOKEN_COMMA) {
		    // nothing to do, should be a number again next
		}
		else if (tok.type == TOKEN_SEMICOLON) {
		    // if this is end of first row, set the row length
		    if (rowlength < 0) rowlength = arrayvals.size();
		
		    // could be an end of line, a number, a closing bracket, or some combination
		    if (!peekNextToken(&tok)) return false;
		    if (!isExpressionStart(tok)) {
			if (!getNextToken(&tok)) return false; // actually consume the token now
			if (tok.type == TOKEN_RBRACKET) {
			    // found end of array
			    break;
			}
			else if (tok.type == TOKEN_CRLF) {
			    // end of line, may be followed by a number or a closing bracket
			    if (!peekNextToken(&tok)) return false;
			    if (!isExpressionStart(tok)) {
				if (!getNextToken(&tok)) return false; // actually consume the token now
				if (tok.type == TOKEN_RBRACKET) {
				    // found end of array
				    break;
				}
				else {
				    logMessage(3, "Parse error in Matlab file: expected number or closing bracket in array assignment");
				}
			    }
			}
			else {
			    logMessage(3, "Parse error in Matlab file: unexpected token type %d (1) in array assignment", tok.type);
			    return false;
			}
		    }
		}
		else if (tok.type == TOKEN_RBRACKET) {
		    // found end
		    break;
		}
		else {
		    logMessage(3, "Parse error in Matlab file: unexpected token type %d (2) in array assignment", tok.type);
		    return false;
		}
	    }
	    if (rowlength < 0) {
		rowlength = arrayvals.size();
	    }
	    val->array.width = rowlength;
	    val->array.height = arrayvals.size() / rowlength;
	    val->array.data = new double[arrayvals.size()];
	    for (i = 0; i < arrayvals.size(); i++) {
		val->array.data[i] = arrayvals[i];
	    }
	    inSpaceSeparatedArray = false;
	    //logMessage(1, "Ending array parse (%d)", arrayvals.size());
	}
	else if (tok.type == TOKEN_STRING) {
	    // string value
	    val->type = CELL_STRING;
	    val->string.value = tok.stringVal;
	}
	else {
	    logMessage(3, "Parse error in Matlab file: expected number or '['");
	    return false;
	}
    }

    // ignore optional transpose operator (for now)
    if (peekNextToken(&tok)) {
	if (tok.type == TOKEN_SINGLEQUOTE) {
	    if (!getNextToken(&tok)) return false;
	}
    }

    return true;
}

void MatlabParser::deepCopyArray(MatlabArray &dest, MatlabArray &src)
{
    int i;
    dest = src;
    dest.data = new double[src.width * src.height];
    for (i = 0; i < (src.width * src.height); i++) {
	dest.data[i] = src.data[i];
    }
}


// parse the entire Matlab file
bool MatlabParser::parse()
{
    MatlabToken tok, tok2;
    int i;
    bool existingArray;
    MatlabArray arr;

    if (!d) return false; // file didn't open at all

    // the lines we're interested in start with '<identifier> = '
    while (true) {
	if (!getNextToken(&tok)) return false;
	if (tok.type == TOKEN_IDENTIFIER) {
	    if (!getNextToken(&tok2)) return false;
	    if (tok2.type == TOKEN_EQUALS) {
		string idname = tok.stringVal;

		if (!peekNextToken(&tok)) return false;
		if (tok.type != TOKEN_LBRACE) {
		    if ((tok.type == TOKEN_IDENTIFIER) && (tok.stringVal == "struct")) {
			// handle structure initialisation
			deleteVariable(idname);
			addStruct(idname);

			// skip the identifier
			if (!getNextToken(&tok)) return false;

			// see if we have () next
			if (!peekNextToken(&tok)) return false;
			if (tok.type == TOKEN_LPAREN) {
			    if (!getNextToken(&tok)) return false;
			    if (tok.type != TOKEN_LPAREN) {
				logMessage(3, "Parse error in Matlab file: expected () after struct keyword");
				return false;
			    }
			    
			    if (!getNextToken(&tok)) return false;
			    if (tok.type != TOKEN_RPAREN) {
				logMessage(3, "Parse error in Matlab file: expected () after struct keyword");
				return false;
			    }
			}
		    }
		    else { // (tok.type != TOKEN_IDENTIFIER) || (tok.stringVal != "struct")
			// standard array or scalar
			MatlabCellContent val;

			if (!parseValue(&val, &existingArray)) return false;
			
			// handle the value returned
			deleteVariable(idname); // delete any existing variable of that name
			
			switch (val.type) {
			case CELL_SCALAR:
			    val.scalar.name = idname;
			    scalars.push_back(val.scalar);
			    break;
			case CELL_ARRAY:
			    if (existingArray) {
				// deep copy if existing array
				deepCopyArray(arr, val.array);
			    }
			    else {
				arr = val.array;
			    }
			    arr.name = idname;
			    arrays.push_back(arr);
			    break;
			case CELL_STRING:
			    val.string.name = idname;
			    strings.push_back(val.string);
			    break;
			}
		    }
		}
		else { // tok.type == TOKEN_LBRACE
		    // cell array
		    vector<MatlabCellContent> cellvals;
		    MatlabCellContent val;
		    int rowlength = -1;

		    if (!getNextToken(&tok)) return false; // skip the opening brace

		    while (true) {
			if (!peekNextToken(&tok)) return false;
			if (tok.type == TOKEN_RBRACE) {
			    // found the end
			    if (!getNextToken(&tok)) return false; // skip the brace
			    break;
			}
			else if (tok.type == TOKEN_CRLF) {
			    // ignore CRLF
			    if (!getNextToken(&tok)) return false;
			}
			else {
			    // should be a value

			    if (!parseValue(&val, &existingArray)) return false;
			    if (existingArray) {
				// if this returns an array that already exists, need
				// to deep copy the array contents
				deepCopyArray(arr, val.array);
				val.array.data = arr.data;
			    }
			    cellvals.push_back(val);
			    
			    // check for semicolon or comma next
			    if (!peekNextToken(&tok)) return false;
			    if (tok.type == TOKEN_COMMA) {
				// skip commas
				if (!getNextToken(&tok)) return false;
			    }
			    else if (tok.type == TOKEN_SEMICOLON) {
				// semicolons start a new line
				if (rowlength < 0) rowlength = cellvals.size();

				if (!getNextToken(&tok)) return false;
			    }
			}
		    }

		    deleteVariable(idname);

		    if (rowlength < 0) rowlength = cellvals.size();
		    MatlabCellArray cellarr;
		    cellarr.name = idname;
		    cellarr.width = rowlength;
		    cellarr.height = cellvals.size() / rowlength;
		    cellarr.data = new MatlabCellContent[cellvals.size()];
		    for (i = 0; i < cellvals.size(); i++) {
			cellarr.data[i] = cellvals[i];
		    }
		    cellarrays.push_back(cellarr);
		}

		// handle end of line, where we should now be
		if (!getNextToken(&tok)) return false;
		if (tok.type == TOKEN_EOF) break;  // break out if reached end of file
		if ((tok.type != TOKEN_CRLF) && (tok.type != TOKEN_SEMICOLON)) {
		    logMessage(3, "Parse error in Matlab file: expected semicolon or end of line after assignment (%s), found %d", idname.c_str(), tok.type);
		    return false;
		}

		// loop back to do next line
		continue;
	    }
	    else {
		string structname, membername;
		int structidx = -1;

		// found an identifier but not an equals after it
		// could be a structure assignment (with or without array index)
		if (tok2.type == TOKEN_DOT) {
		    // assignment without index
		    structidx = 1;
		    structname = tok.stringVal;

		    if (!getNextToken(&tok)) return false;
		    if (tok.type != TOKEN_IDENTIFIER) {
			logMessage(3, "Parse error in Matlab file: expected structure member name");
			return false;
		    }
		    membername = tok.stringVal;
		}
		else if (tok2.type == TOKEN_LPAREN) {
		    // assignment with index
		    structname = tok.stringVal;

		    if (!getNextToken(&tok)) return false;
		    if (tok.type != TOKEN_NUMBER) {
			logMessage(3, "Parse error in Matlab file: expected numeric index");
			return false;
		    }
		    structidx = tok.numVal;

		    if (!getNextToken(&tok)) return false;
		    if (tok.type != TOKEN_RPAREN) {
			logMessage(3, "Parse error in Matlab file: expected right parenthesis");
			return false;
		    }

		    if (!getNextToken(&tok)) return false;
		    if (tok.type != TOKEN_DOT) {
			logMessage(3, "Parse error in Matlab file: expected dot");
			return false;
		    }

		    if (!getNextToken(&tok)) return false;
		    if (tok.type != TOKEN_IDENTIFIER) {
			logMessage(3, "Parse error in Matlab file: expected structure member name");
			return false;
		    }
		    membername = tok.stringVal;
		}

		if (structidx >= 0) {
		    // check for equals
		    if (!getNextToken(&tok)) return false;
		    if (tok.type != TOKEN_EQUALS) {
			logMessage(3, "Parse error in Matlab file: expected equals");
			return false;
		    }

		    // parse the value
		    MatlabCellContent val;
		    if (!parseValue(&val, &existingArray)) return false;

		    // if this returns an array that already exists, need to deep copy the
		    // array's contents
		    if (existingArray) {
			deepCopyArray(arr, val.array);
			val.array.data = arr.data;
		    }

		    // do struct member assignment
		    // first find the structure in question
		    for (i = 0; i < structs.size(); i++) {
			if (structs[i].name == structname) break;
		    }
		    if (i == structs.size()) {
			// add new structure
			addStruct(structname);
		    }
		    MatlabStruct *str = &structs[i];

		    // now ensure the index referenced exists
		    MatlabStructElement mse;
		    while (str->elements.size() < structidx) {
			str->elements.push_back(mse);
		    }

                    // convert to zero-based
                    structidx--;

		    // look for the member referenced
		    for (i = 0; i < str->elements[structidx].memberNames.size(); i++) {
			if (str->elements[structidx].memberNames[i] == membername) break;
		    }
		    if (i == str->elements[structidx].memberNames.size()) {
			// add new member
			str->elements[structidx].memberNames.push_back(membername);
			str->elements[structidx].memberValues.push_back(val);
		    }
		    else {
			// update existing member
			str->elements[structidx].memberValues[i] = val;
		    }

		    // check for end of line
		    if (!getNextToken(&tok)) return false;
		    if (tok.type == TOKEN_EOF) break;  // break out if reached end of file
		    if ((tok.type != TOKEN_CRLF) && (tok.type != TOKEN_SEMICOLON)) {
			logMessage(3, "Parse error in Matlab file: expected semicolon or end of line after assignment (%s), found %d", structname.c_str(), tok.type);
			return false;
		    }

		    // loop back to do next line
		    continue;
		}
	    }
	}
	// skip to end of line
	while ((tok.type != TOKEN_CRLF) && (tok.type != TOKEN_EOF)) {
	    if (!getNextToken(&tok)) return false;
	}
	// break out of loop if end of file found
	if (tok.type == TOKEN_EOF) break;
    }
    return true;
}
