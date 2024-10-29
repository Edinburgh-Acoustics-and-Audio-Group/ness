/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2015. All rights reserved.
 *
 * Singleton class to hold global settings used by entire simulation
 */
#ifndef _GLOBALSETTINGS_H_
#define _GLOBALSETTINGS_H_

#include <cstdio>
using namespace std;

class GlobalSettings {
 public:
    // get singleton instance
    static GlobalSettings *getInstance() { 
	if (instance == NULL) instance = new GlobalSettings();
	return instance;
    }

    void setSampleRate(double sr) { 
	sampleRate = sr;
	// k is inverse of sampleRate
	k = 1.0 / sampleRate;
    }
    void setDuration(double d) { duration = d; }
    void setK(double kk) { k = kk; }

    void setEnergyOn(bool eo) { 
	energyOn = eo; 
	if (energyOn) {
	    strikesOn = false;
	    bowingOn = false;
	    lossMode = 0;
	    gpuEnable = false;
	}
    }
    void setStrikesOn(bool so) { strikesOn = so; }
    void setBowingOn(bool bo) { bowingOn = bo; }
    void setHighPassOn(bool hpo) { highPassOn = hpo; }

    void setSymmetricSolve(bool sym) { symmetricSolve = sym; }
    void setIterinv(int ii) { iterinv = ii; }
    void setPcgTolerance(double tol) { pcgTolerance = tol; }
    void setTestStrike(bool ts) { testStrike = ts; }
    void setNoRecalcQ(bool nrq) { noRecalcQ = nrq; }
    void setLossMode(int lm) { lossMode = lm; }

    void setFixPar(double fp) { fixpar = fp; }

    void setLinear(bool l) { linear = l; }

    void setMaxThreads(int mt) { maxThreads = mt; }

    void setPcgMaxIterations(int pmi) { pcgMaxIterations = pmi; }
    void setInterpolateInputs(bool ii) { interpolateInputs = ii; }
    void setInterpolateOutputs(bool io) { interpolateOutputs = io; }
    void setCuda2dBlockW(int cbw) { cuda2dBlockW = cbw; }
    void setCuda2dBlockH(int cbh) { cuda2dBlockH = cbh; }
    void setCuda3dBlockW(int cbw) { cuda3dBlockW = cbw; }
    void setCuda3dBlockH(int cbh) { cuda3dBlockH = cbh; }
    void setCuda3dBlockD(int cbd) { cuda3dBlockD = cbd; }
    void setNegateInputs(bool ni) { negateInputs = ni; }
    void setLogState(bool ls) { logState = ls; }
    void setLogMatrices(bool lm) { logMatrices = lm; }
    void setGpuEnabled(bool ge) { gpuEnable = ge; }
    void setEstimate(bool e) { estimate = e; }

    void setAVXEnabled(bool a) { avx = a; }

    void setMaxOut(double mo) { maxOut = mo; }

    void setImpulse(double i) { impulse = i; }

    void setNormaliseOuts(bool no) { normaliseOuts = no; }
    void setResampleOuts(bool ro) { resampleOuts = ro; }


    double getSampleRate() { return sampleRate; }
    double getDuration() { return duration; }
    double getK() { return k; }

    int getNumTimesteps() {
	return (int)(sampleRate * duration);
    }

    bool getEnergyOn() { return energyOn; }
    bool getStrikesOn() { return strikesOn; }
    bool getBowingOn() { return bowingOn; }
    bool getHighPassOn() { return highPassOn; }

    bool getSymmetricSolve() { return symmetricSolve; }
    int getIterinv() { return iterinv; }
    double getPcgTolerance() { return pcgTolerance; }
    bool getTestStrike() { return testStrike; }
    bool getNoRecalcQ() { return noRecalcQ; }
    int getLossMode() { return lossMode; }

    double getFixPar() { return fixpar; }

    bool getLinear() { return linear; }

    int getMaxThreads() { return maxThreads; }

    int getPcgMaxIterations() { return pcgMaxIterations; }
    bool getInterpolateInputs() { return interpolateInputs; }
    bool getInterpolateOutputs() { return interpolateOutputs; }
    int getCuda2dBlockW() { return cuda2dBlockW; }
    int getCuda2dBlockH() { return cuda2dBlockH; }
    int getCuda3dBlockW() { return cuda3dBlockW; }
    int getCuda3dBlockH() { return cuda3dBlockH; }
    int getCuda3dBlockD() { return cuda3dBlockD; }
    bool getNegateInputs() { return negateInputs; }
    bool getLogState() { return logState; }
    bool getLogMatrices() { return logMatrices; }
    bool getGpuEnabled() { return gpuEnable; }
    bool getEstimate() { return estimate; }

    bool getAVXEnabled() { return avx; }

    double getMaxOut() { return maxOut; }

    bool getImpulse() { return impulse; }

    bool getNormaliseOuts() { return normaliseOuts; }

    bool getResampleOuts() { return resampleOuts; }

 private:
    static GlobalSettings *instance;
    GlobalSettings();
    ~GlobalSettings();

    double sampleRate;
    double duration;
    double k;
    
    bool energyOn;
    bool strikesOn;   // FIXME: are these two actually used anywhere?
    bool bowingOn;
    bool highPassOn;

    bool symmetricSolve;
    int iterinv;
    double pcgTolerance;
    bool testStrike;
    bool noRecalcQ;
    int lossMode;

    double fixpar;

    bool linear;

    int maxThreads;


    int pcgMaxIterations;
    bool interpolateInputs;
    bool interpolateOutputs;
    int cuda2dBlockW;
    int cuda2dBlockH;
    int cuda3dBlockW;
    int cuda3dBlockH;
    int cuda3dBlockD;
    bool negateInputs;
    bool logState;
    bool logMatrices;
    bool gpuEnable;

    bool estimate;

    bool avx;

    bool impulse;

    double maxOut;

    bool normaliseOuts;

    bool resampleOuts;
};

#endif
