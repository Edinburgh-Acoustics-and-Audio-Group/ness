/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */
#include "Airbox.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"

#include <cmath>
using namespace std;

Airbox::Airbox(string name)
    : Component3D(name)
{
}

Airbox::~Airbox()
{
}

void Airbox::setup(double lx, double ly, double lz, double c_a, double rho_a, double tau1)
{
    double NPX, NPY, NPZ;

    logMessage(1, "Entering airbox setup: %f, %f, %f, %f, %f", lx, ly, lz, c_a, rho_a);

    LX = lx;
    LY = ly;
    LZ = lz;

    this->rho_a = rho_a;
    this->tau1 = tau1;
    this->c_a = c_a;

    double SR = GlobalSettings::getInstance()->getSampleRate();

    k = 1.0 / ((double)SR);
    Q = sqrt(3.0 * c_a * c_a * k * k + 12.0 * tau1 * k);
    NPX = floor(LX / Q);
    if (!SettingsManager::getInstance()->getBoolSetting(name, "no_recalc_q")) {
	Q = LX / NPX;
    }
    NPY = floor(LY / Q);
    NPZ = floor(LZ / Q);
    Gamma = (k * c_a) / Q;
    lambda = sqrt(2.0 * tau1 * k / (Q*Q));
    gammabar = sqrt(Gamma*Gamma + lambda*lambda);

    allocateState((int)NPX+1, (int)NPY+1, (int)NPZ+1);

    // FIXME: may not be correct
    alpha = (k*k) / (rho_a * Q * Q);
    bowFactor = k / (2.0 * Q * Q);

    logMessage(1, "Airbox scalars: %f, %f, %f, %d, %d, %d, %d", k, Q, Gamma, nx, ny, nz, ss);
}

int Airbox::getGPUScore()
{
    if (SettingsManager::getInstance()->getBoolSetting(name, "disable_gpu")) return GPU_SCORE_NO;
    return GPU_SCORE_GREAT;
}

void Airbox::addPlate(int zb, double *true_Psi)
{
    // only needed for energy check, not implemented here
}

