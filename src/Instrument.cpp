/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "Instrument.h"
#include "Logger.h"
#include "WavWriter.h"
#include "GlobalSettings.h"
#include "SettingsManager.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
using namespace std;

Instrument::Instrument()
{
    profiler = NULL;
#ifndef NO_THREADS
    workerThreads = NULL;
#endif
}

Instrument::~Instrument()
{
    int i;
    for (i = 0; i < components.size(); i++) {
	delete components[i];
    }
    for (i = 0; i < outputs.size(); i++) {
	delete outputs[i];
    }
    for (i = 0; i < connections.size(); i++) {
	delete connections[i];
    }

    for (i = 0; i < serialTasks.size(); i++) {
	delete serialTasks[i];
    }
    for (i = 0; i < parallelTasks.size(); i++) {
	delete parallelTasks[i];
    }

    for (i = 0; i < serialConnectionTasks.size(); i++) {
	delete serialConnectionTasks[i];
    }
    for (i = 0; i < parallelConnectionTasks.size(); i++) {
	delete parallelConnectionTasks[i];
    }

    if (profiler) {
	delete profiler;
    }
}

void Instrument::runTimestep(int n)
{
    int i;

    if (!profiler) {
	profiler = new Profiler(components.size() + connections.size());
    }

#ifndef NO_THREADS
    if (workerThreads == NULL) {
#endif
	// run all components on single thread
	for (i = 0; i < components.size(); i++) {
	    profiler->start(i);
	    components[i]->runTimestep(n);
	    profiler->end(i);
	}

	// and connections
	for (i = 0; i < connections.size(); i++) {
	    profiler->start(components.size() + i);
	    connections[i]->runTimestep(n);
	    profiler->end(components.size() + i);
	}
#ifndef NO_THREADS
    }
    else {
#ifdef USE_GPU
	// run GPU components first
	for (i = 0; i < gpuComponents.size(); i++) {
	    gpuComponents[i]->runTimestep(n);
	}
#endif

	// multi-threaded component run
	// run the parallel tasks first
	WorkerThread::setGo();   // tell the threads to run
	WorkerThread::workerThreadRun(0, n); // run our own task(s)
	WorkerThread::clearGo(); // clear the "go" signal
	//logMessage(1, "Finished running parallel components");

	// now run the serial tasks
	for (i = 0; i < serialTasks.size(); i++) {
	    serialTasks[i]->runTimestep(n);
	}

	// run the connection tasks
	// parallel ones first, if any
	if (parallelConnectionTasks.size() > 0) {
	    WorkerThread::connectionsGo();
	    WorkerThread::workerThreadConnectionRun(0, n);
	    WorkerThread::clearGo();
	}

	// now serial ones
	for (i = 0; i < serialConnectionTasks.size(); i++) {
	    serialConnectionTasks[i]->runTimestep(n);
	}
    }
#endif
}

void Instrument::endTimestep(int n)
{
    int i;
    for (i = 0; i < outputs.size(); i++) {
	outputs[i]->runTimestep(n);
    }

    for (i = 0; i < components.size(); i++) {
	components[i]->swapBuffers(n);
    }
}

double Instrument::getMaxOutput()
{
    int i, j;
    double max;
    double curr;

    max = 0.0;
    // get absolute maximum across entire output space
    for (i = 0; i < GlobalSettings::getInstance()->getNumTimesteps(); i++) {
	curr = 0.0;
	for (j = 0; j < outputs.size(); j++) {
	    curr += outputs[j]->getData()[i];
	}
	curr = fabs(curr);
	if (curr > max) max = curr;
    }
    logMessage(1, "Max output is %f", max);
    return max;
}

void Instrument::saveOutputs(string outputname, bool individual, bool raw)
{
    int i, j;
    double max;
    GlobalSettings *gs = GlobalSettings::getInstance();
    
#ifdef USE_GPU
    // copy outputs back from GPU
    for (i = 0; i < outputs.size(); i++) {
	outputs[i]->copyFromGPU();
    }
#endif

    // write raw outputs if requested
    if (raw) {
	logMessage(1, "Saving raw output data");
	for (i = 0; i < outputs.size(); i++) {
	    ostringstream convert;
	    convert << (i+1);
	    string nstr = convert.str();
	    string compname = outputs[i]->getComponent()->getName();
	    string filename = outputname + "-" + compname + "-" + nstr + ".f64";
	    if (outputs.size() == 1) filename = outputname + ".f64";
	    outputs[i]->saveRawData(filename);
	}
    }
    
    // then high-pass filter the channels if requested
    if (gs->getHighPassOn()) {
	logMessage(1, "Applying high-pass filter");
	for (i = 0; i < outputs.size(); i++) {
	    outputs[i]->highPassFilter();
	}
    }

    // normalise the channels if requested
    if (gs->getNormaliseOuts()) {
	logMessage(1, "Normalising output channels");
	for (i = 0; i < outputs.size(); i++) {
	    outputs[i]->normalise();
	}
    }

    // always write stereo mix (if there's more than one channel to mix!)
    if (outputs.size() > 1) {
	logMessage(1, "Writing stereo mix output");
	max = getMaxOutput();
	WavWriter mix(outputname + "-mix.wav");
	if (!mix.writeStereoMix(&outputs, max)) {
	    logMessage(5, "Error writing stereo mix output!");
	}
    }

    // write individual output files if requested
    if (individual) {
	logMessage(1, "Writing individual channel wavs");
	for (i = 0; i < outputs.size(); i++) {
	    ostringstream convert;
	    convert << (i+1);
	    string nstr = convert.str();
	    string compname = outputs[i]->getComponent()->getName();
	    string filename = outputname + "-" + compname + "-" + nstr + ".wav";
	    if (outputs.size() == 1) filename = outputname + ".wav";
	    WavWriter channel(filename);
	    max = outputs[i]->getMaxValue();
	    if (!channel.writeMonoWavFile(outputs[i], max)) {
		logMessage(5, "Error writing wav file for channel %d!", i+1);
	    }
	}
    }

    if (gs->getEnergyOn()) {
	// write energy values
	double *ce;
	int n = gs->getNumTimesteps();
	double *energy = new double[n];
	for (i = 0; i < n; i++) {
	    energy[i] = 0;
	}
	// add component energies
	for (i = 0; i < components.size(); i++) {
	    ce = components[i]->getEnergy();
	    if (ce != NULL) {
		for (j = 0; j < n; j++) {
		    energy[j] += ce[j];
		}
	    }
	}
	// add connection energies
	for (i = 0; i < connections.size(); i++) {
	    ce = connections[i]->getEnergy();
	    if (ce != NULL) {
		for (j = 0; j < n; j++) {
		    energy[j] += ce[j];
		}
	    }
	}
	// save energy values
	ofstream of("energy.txt");
	if (of.good()) {
	    of.precision(20);
	    for (i = 0; i < n; i++) {
		of << energy[i] << endl;
	    }
	    of.close();
	}
	else {
	    logMessage(5, "Error writing energy values to file energy.txt");
	}
	delete[] energy;
    }
}

Component *Instrument::getComponentByName(string name)
{
    int i;
    for (i = 0; i < components.size(); i++) {
	if (name == components[i]->getName()) return components[i];
    }
    return NULL;
}

// called after main initialisation, but before the main loop starts
void Instrument::optimise()
{
    int i;

    // log matrices if enabled
    for (i = 0; i < components.size(); i++) {
	if (SettingsManager::getInstance()->getBoolSetting(components[i]->getName(), "log_matrices")) {
	    components[i]->logMatrices();
	}
    }

    serialTasks.clear();
    parallelTasks.clear();
    gpuComponents.clear();

    // first do GPU optimisation
#ifdef USE_GPU
    if (GlobalSettings::getInstance()->getGpuEnabled()) {
	for (i = 0; i < components.size(); i++) {
	    // look for great GPU components
	    if (components[i]->getGPUScore() == GPU_SCORE_GREAT) {
		// try to move to GPU
		if (components[i]->moveToGPU()) {
		    gpuComponents.push_back(components[i]);
		}
	    }
	}
	for (i = 0; i < components.size(); i++) {
	    // now look for good GPU components
	    if (components[i]->getGPUScore() == GPU_SCORE_GOOD) {
		// try to move to GPU
		if (components[i]->moveToGPU()) {
		    gpuComponents.push_back(components[i]);
		}
	    }
	}
	// notify outputs and connections that we've moved components
	// to GPU
	for (i = 0; i < connections.size(); i++) {
	    connections[i]->maybeMoveToGPU();
	}
	for (i = 0; i < outputs.size(); i++) {
	    outputs[i]->maybeMoveToGPU();
	}
    }
#endif

#ifndef NO_THREADS
    // only one thread allowed, don't optimise
    if (GlobalSettings::getInstance()->getMaxThreads() <= 1) return;

    // get serial and parallel tasks from components
    for (i = 0; i < components.size(); i++) {
	if (components[i]->isOnGPU()) continue;

	components[i]->getParallelTasks(parallelTasks);
	components[i]->getSerialTasks(serialTasks);
    }

    logMessage(1, "%d parallel tasks, %d serial, %d components on GPU",
	       parallelTasks.size(), serialTasks.size(), gpuComponents.size());

    // get serial and parallel tasks from connections
    for (i = 0; i < connections.size(); i++) {
	if (parallelTasks.size() > 1) {
	    connections[i]->getParallelTasks(parallelConnectionTasks);
	}
	else {
	    // if there's going to be no worker threads, will have to run them
	    // all in serial
	    connections[i]->getParallelTasks(serialConnectionTasks);
	}
	connections[i]->getSerialTasks(serialConnectionTasks);
    }

    logMessage(1, "%d parallel connection tasks, %d serial",
	       parallelConnectionTasks.size(), serialConnectionTasks.size());

    // create worker threads for the parallel tasks
    if (parallelTasks.size() > 1) {
	workerThreads = WorkerThread::createWorkerThreads(parallelTasks,
							  parallelConnectionTasks);
    }

#endif
}
