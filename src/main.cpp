/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Top level code, command line parsing, main loop
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <fstream>
using namespace std;

#ifdef WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "Instrument.h"
#include "InstrumentParser.h"
#include "ScoreParser.h"
#include "MaterialsManager.h"
#include "Input.h"

#include "GlobalSettings.h"
#include "SettingsManager.h"
#include "Logger.h"
#include "Profiler.h"

#define FRAMEWORK_VERSION "0.1.1"

#define BRASS_VERSION "1.0"

#define GUITAR_VERSION "1.0"

static Instrument *instrument;

static string instrumentName = "instrument.txt";
static string scoreName = "score.txt";

static string outputName = "output";
static bool saveRawOutputs = false;
static bool saveIndividualOutputs = true;

// name of executable, not including any path components
static string exeName;

static void getExecutableLocation(char *buf, int bufsize)
{
#ifndef WIN32
    size_t sz = readlink("/proc/self/exe", buf, bufsize);
    buf[sz] = 0;
    char *slash = strrchr(buf, '/');
    if (slash) *slash = 0;
#else
    GetModuleFileNameA(NULL, buf, bufsize);
    char *slash = strrchr(buf, '\\');
    if (slash) *slash = 0;
#endif
}

void getCacheDirectory(char *buf, int bufsize)
{
#ifndef WIN32
    const char *home = getenv("HOME");
    if (home == NULL) {
	buf[0] = 0;
	return;
    }
    sprintf(buf, "%s/.ness", home);
    mkdir(buf, 0755);
    strcat(buf, "/cache/");
    mkdir(buf, 0755);
#else
    char mydoc[MAX_PATH];
    SHGetFolderPathA(NULL, CSIDL_PERSONAL, NULL, 0, mydoc);
    sprintf(buf, "%s\\.ness", mydoc);
	CreateDirectoryA(buf, NULL);
    strcat(buf, "\\cache\\");
	CreateDirectoryA(buf, NULL);
#endif
}


static void detectCPU()
{
    GlobalSettings *gs = GlobalSettings::getInstance();
#if defined USE_AVX && (defined __amd64__ || defined __i386__)
    unsigned int ecx, edx;
    int code = 1;
    asm volatile("cpuid" : "=c"(ecx), "=d"(edx) : "a"(code) : "ebx");
    if (ecx & (1 << 28)) {
	logMessage(1, "CPU supports AVX");
	gs->setAVXEnabled(true);
    }
    else {
	logMessage(1, "CPU does not support AVX");
	gs->setAVXEnabled(false);
    }    
#elif defined WIN32
	if (IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE)) {
		logMessage(1, "CPU supports AVX");
		gs->setAVXEnabled(true);
	}
	else {
		logMessage(1, "CPU does not support AVX");
		gs->setAVXEnabled(false);
	}
#else
    gs->setAVXEnabled(false);
#endif
}


/*
 * Load instrument and score
 *
 * Returns 0 on failure
 */
static int initialise()
{
    detectCPU();

#ifndef BRASS_ONLY 
#ifndef GUITAR_ONLY
    // assume that materials database is in directory with executable
    char matfile[1000];
#ifndef __APPLE__
    getExecutableLocation(matfile, 1000);
    strcat(matfile, "/materials.txt");
#else
    strcpy(matfile, "materials.txt");
#endif
    MaterialsManager::getInstance()->loadMaterials(matfile);
#endif
#endif

    // get simulation length from score file first!
    double duration = ScoreParser::getDuration(scoreName, exeName);
    if (duration < 0.0) {
	logMessage(5, "Unable to get duration from score file!");
	return 0;
    }
    GlobalSettings::getInstance()->setDuration(duration);

    instrument = InstrumentParser::parseInstrument(instrumentName, exeName);
    if (!instrument) {
	logMessage(5, "Unable to parse instrument file");
	return 0;
    }

    if (!ScoreParser::parseScore(scoreName, instrument, exeName)) {
	logMessage(5, "Unable to parse score file");
	delete instrument;
	return 0;
    }

    // allow instrument to setup for multicore/CUDA
    instrument->optimise();

    if (GlobalSettings::getInstance()->getEnergyOn()) {
	Input::setFirstInputTimestep(0);
    }

    return 1;
}

/*
 * Run the main simulation loop
 */
static void mainLoop()
{
    int NF = GlobalSettings::getInstance()->getNumTimesteps();
    int n;
    int start;

    if (GlobalSettings::getInstance()->getEstimate()) {
	if (NF > 1000) {
	    NF = 1000;
	}
	start = 0;
    }
    else {
	// start from when first input happens. everything before
	// that is all zeros
	start = Input::getFirstInputTimestep();
	logMessage(1, "Starting from timestep %d", start);
    }

    for (n = start; n < NF; n++) {
	if ((n % 1000) == 0) {
	    logMessage(1, "Iteration %d", n);

	    // write progress to a file every 1000 iterations, allowing the frontend
	    // to monitor long running simulations
	    ofstream of("progress.txt");
	    of << n << endl;
	    of.close();
	}
	instrument->runTimestep(n);
	instrument->endTimestep(n);
    }
}

/*
 * Save outputs and free resources
 */
static void finalise()
{
    if (!GlobalSettings::getInstance()->getEstimate()) {
	instrument->saveOutputs(outputName, saveIndividualOutputs, saveRawOutputs);
    }
    delete instrument;
}


/*
 * Parse the command line and configure the code (mostly via GlobalSettings)
 *
 * Returns 0 on failure (invalid command line etc.)
 */
static int parseCommandLine(int argc, char *argv[])
{
    int i;
    GlobalSettings *gs = GlobalSettings::getInstance();
    SettingsManager *sm = SettingsManager::getInstance();

    for (i = 1; i < argc; i++) {
	if (strchr(argv[i], ':')) {
	    // component-specific setting
	    if (i < (argc-1)) {
		if (sm->putSetting(argv[i], argv[i+1])) i++;
	    }
	    else {
		sm->putSetting(argv[i]);
	    }
	    continue;
	}

	if (!strcmp(argv[i], "-i")) {
	    // instrument filename
	    if (i == (argc-1)) return 0;
	    i++;
	    instrumentName = argv[i];
	}
	else if (!strcmp(argv[i], "-s")) {
	    // score filename
	    if (i == (argc-1)) return 0;
	    i++;
	    scoreName = argv[i];
	}
	else if (!strcmp(argv[i], "-v")) {
	    // get version
#ifndef BRASS_ONLY
#ifndef GUITAR_ONLY
	    cout << "NeSS Framework, version " << FRAMEWORK_VERSION << endl;
#else
	    cout << "NeSS Guitar and Net Code, version " << GUITAR_VERSION << endl;
#endif
#else
	    cout << "NeSS Brass Code, version " << BRASS_VERSION << endl;
#endif
	    exit(0);
	}
	else if (!strcmp(argv[i], "-r")) {
	    // save raw outputs
	    saveRawOutputs = true;
	}
	else if (!strcmp(argv[i], "-e")) {
	    // estimate
	    gs->setEstimate(true);
	}
	else if (!strcmp(argv[i], "-o")) {
	    // output base name
	    if (i == (argc-1)) return 0;
	    i++;
	    outputName = argv[i];	    
	}
	else if (!strcmp(argv[i], "-energy")) {
	    // enable energy check
	    gs->setEnergyOn(true);
	}
	else if (!strcmp(argv[i], "-impulse")) {
	    // enable impulse
	    gs->setImpulse(true);
	}
	else if ((!strcmp(argv[i], "-normalise_outs")) ||
		 (!strcmp(argv[i], "-normalize_outs"))) {
	    // enable normalising outputs before stereo mix
	    gs->setNormaliseOuts(true);
	}
#ifndef BRASS_ONLY
	else if (!strcmp(argv[i], "-c")) {
	    // save only stereo mix or individual outputs as well
	    if (i == (argc-1)) return 0;
	    i++;
	    if (!strcmp(argv[i], "stereo")) {
		saveIndividualOutputs = false;
	    }
	    else if (!strcmp(argv[i], "all")) {
		saveIndividualOutputs = true;
	    }
	    else return 0;
	}

#ifndef GUITAR_ONLY
	else if (!strcmp(argv[i], "-symmetric")) {
	    gs->setSymmetricSolve(true);
	}
	else if (!strcmp(argv[i], "-iterinv")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setIterinv(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-pcg_tol")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setPcgTolerance(atof(argv[i]));
	}
	else if (!strcmp(argv[i], "-no_recalc_q")) {
	    gs->setNoRecalcQ(true);
	}
	else if (!strcmp(argv[i], "-loss_mode")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setLossMode(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-fixpar")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setFixPar(atof(argv[i]));
	}
	else if (!strcmp(argv[i], "-linear")) {
	    gs->setLinear(true);
	}
	else if (!strcmp(argv[i], "-pcg_max_it")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setPcgMaxIterations(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-disable_gpu")) {
	    gs->setGpuEnabled(false);
	}

	else if (!strcmp(argv[i], "-cuda_2d_block_w")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setCuda2dBlockW(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-cuda_2d_block_h")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setCuda2dBlockH(atoi(argv[i]));
	}

	else if (!strcmp(argv[i], "-cuda_3d_block_w")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setCuda3dBlockW(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-cuda_3d_block_h")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setCuda3dBlockH(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-cuda_3d_block_d")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setCuda3dBlockD(atoi(argv[i]));
	}
#endif

	else if (!strcmp(argv[i], "-max_threads")) {
	    if (i == (argc-1)) return 0;
	    i++;
	    gs->setMaxThreads(atoi(argv[i]));
	}
	else if (!strcmp(argv[i], "-interpolate_inputs")) {
	    gs->setInterpolateInputs(true);
	}
	else if (!strcmp(argv[i], "-interpolate_outputs")) {
	    gs->setInterpolateOutputs(true);
	}
	else if (!strcmp(argv[i], "-negate_inputs")) {
	    gs->setNegateInputs(true);
	}
	else if (!strcmp(argv[i], "-log_state")) {
	    gs->setLogState(true);
	}
	else if (!strcmp(argv[i], "-log_matrices")) {
	    gs->setLogMatrices(true);
	}
#endif

	else {
	    return 0;
	}
    }
    return 1;
}

static void usage(char *progname)
{
#ifndef BRASS_ONLY
#ifndef GUITAR_ONLY
    fprintf(stderr, "Usage: %s [-v] [-i <instrument file>] [-s <score file>] [-o <output filename base>] [-c stereo|all] [-r] [-e] [-energy] [-impulse] [-normalise_outs|-normalize_outs] [-symmetric] [-iterinv <count>] [-pcg_tol <tolerance>] [-no_recalc_q] [-loss_mode <mode>] [-fixpar] [-linear] [-max_threads <count>] [-pcg_max_it <count>] [-interpolate_inputs] [-interpolate_outputs] [-negate_inputs] [-log_state] [-log_matrices] [-disable_gpu] [-cuda_2d_block_w <size>] [-cuda_2d_block_h <size>] [-cuda_3d_block_w <size>] [-cuda_3d_block_h <size>] [-cuda_3d_block_d <size>]\n", progname);
#else
    fprintf(stderr, "Usage: %s [-v] [-i <instrument file>] [-s <score file>] [-o <output filename base>] [-c stereo|all] [-r] [-e] [-energy] [-impulse] [-normalise_outs|-normalize_outs] [-max_threads <count>] [-interpolate_inputs] [-interpolate_outputs] [-negate_inputs] [-log_state] [-log_matrices]\n", progname);
#endif
#else
    fprintf(stderr, "Usage: %s [-v] [-i <instrument file>] [-s <score file>] [-o <output filename>] [-r] [-e] [-energy] [-impulse]\n", progname);
#endif
}

static const char *profileNames[] = {
    "startup", "main loop", "finalise"
};

int main(int argc, char *argv[])
{
    if (!parseCommandLine(argc, argv)) {
	usage(argv[0]);
	return 1;
    }

    // get executable name
    exeName = argv[0];
    int lastslash = exeName.rfind("/");
    if (lastslash != string::npos) {
	exeName = exeName.substr(lastslash + 1);
    }
    logMessage(1, "Executable name: %s", exeName.c_str());

    Profiler profiler(3, profileNames);

    profiler.start(0);
    if (!initialise()) {
	return 1;
    }
    profiler.end(0);

    profiler.start(1);
    mainLoop();
    profiler.end(1);

    if (GlobalSettings::getInstance()->getEstimate()) {
	double simthousandtime = profiler.get(1);
	double simestimatetime = profiler.get(0) + simthousandtime * GlobalSettings::getInstance()->getNumTimesteps() / 1000.0;
	logMessage(5, "Simulation time for 1000 timesteps : %fs", simthousandtime);
	logMessage(5, "Estimated time for full run : %fs", simestimatetime);	
    }

    profiler.start(2);
    finalise();
    profiler.end(2);

    // uncomment to see profile
    //printf("Profile: %s\n", profiler.print().c_str());

    return 0;
}
