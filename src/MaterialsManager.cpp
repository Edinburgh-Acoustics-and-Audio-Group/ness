/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 */

#include "MaterialsManager.h"
#include "Logger.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

MaterialsManager *MaterialsManager::instance = NULL;

MaterialsManager::MaterialsManager()
{
    materials = new vector<Material*>();
}

MaterialsManager::~MaterialsManager()
{
    int i;
    for (i = 0; i < materials->size(); i++) {
	Material *mat = materials->at(i);
	delete mat;
    }
    delete materials;
}

void MaterialsManager::loadMaterials(string filename)
{
    logMessage(3, "Loading materials from %s", filename.c_str());
    ifstream in(filename.c_str());

    if (!in.good()) {
	logMessage(5, "Error! Cannot open materials file %s", filename.c_str());
	return;
    }

    while (in.good()) {
	char linebuf[1000];
	in.getline(linebuf, 1000);
	
	// ignore comments
	char *hash = strchr(linebuf, '#');
	if (hash) *hash = 0;
	stringstream ss(linebuf, stringstream::in);

	string matname;
	double ym, pr, d;
	ss >> matname;
	if (!ss.good()) continue;
	ss >> ym >> pr >> d;

	Material *mat = new Material(matname, ym, pr, d);
	materials->push_back(mat);
    }
    in.close();
}

Material *MaterialsManager::getMaterial(string name)
{
    int i;
    // would be more efficient with a hashtable, but this is just startup code and we
    // may never have enough materials for the time to be significant
    for (i = 0; i < materials->size(); i++) {
	Material *mat = materials->at(i);
	if (mat->getName() == name) {
	    return mat;
	}
    }
    logMessage(3, "getMaterial failed for %s", name.c_str());
    return NULL;
}
