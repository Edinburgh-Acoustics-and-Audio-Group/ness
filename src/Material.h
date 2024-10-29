/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Represents a material that components can be made from
 */
#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include <string>
using namespace std;

class Material {
 public:
    Material(string nm, double ym, double pr, double dy);
    virtual ~Material();

    string getName() { return name; }
    double getYoungsModulus() { return youngsModulus; }
    double getPoissonsRatio() { return poissonsRatio; }
    double getDensity() { return density; }

 private:
    string name;

    double youngsModulus;
    double poissonsRatio;
    double density;
};

#endif
