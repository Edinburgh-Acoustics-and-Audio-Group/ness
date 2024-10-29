/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Singleton that manages all materials in the system
 */
#ifndef _MATERIALSMANAGER_H_
#define _MATERIALSMANAGER_H_

#include "Material.h"

#include <vector>
#include <string>
using namespace std;

class MaterialsManager {
 public:
    // get singleton
    static MaterialsManager *getInstance() {
	if (instance == NULL) {
	    instance = new MaterialsManager();
	}
	return instance;
    }

    // load materials from text file specified
    void loadMaterials(string filename);

    // look up material by name
    Material *getMaterial(string name);

 private:
    MaterialsManager();
    virtual ~MaterialsManager();

    vector<Material*> *materials;
    static MaterialsManager *instance;
};

#endif
