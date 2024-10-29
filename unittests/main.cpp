/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * Top level unit test runner
 */

#include "TestMaterial.h"
#include "TestMaterialsManager.h"
#include "TestGlobalSettings.h"
#include "TestLogger.h"
#include "TestCsrmatrix.h"
#include "TestBanded.h"
#include "TestPcg.h"

#include "TestAirbox.h"
#include "TestAirboxIndexed.h"
#include "TestBar.h"
#include "TestComponent.h"
#include "TestComponent1D.h"
#include "TestComponent2D.h"
#include "TestComponent3D.h"
#include "TestComponentString.h"
#include "TestPlate.h"
#include "TestPlateEmbedded.h"
#include "TestSoundBoard.h"
#include "TestStringWithFrets.h"

#include "TestInput.h"
#include "TestInputBow.h"
#include "TestInputPluck.h"
#include "TestInputSample.h"
#include "TestInputStrike.h"
#include "TestInputWav.h"

#include "TestOutput.h"
#include "TestOutputDifference.h"
#include "TestOutputPressure.h"

#include "TestConnectionP2P.h"
#include "TestConnectionZero.h"
#include "TestConnectionZeroPt1.h"
#include "TestEmbedding.h"

#include "TestParser.h"
#include "TestInstrumentParserMP3D.h"
#include "TestInstrumentParserSoundboard.h"
#include "TestInstrumentParserXML.h"
#include "TestInstrumentParserZero.h"
#include "TestInstrumentParserZeroPt1.h"
#include "TestScoreParserSoundboard.h"
#include "TestScoreParserXML.h"
#include "TestScoreParserZero.h"

#include "TestInstrument.h"
#include "TestMathUtil.h"
#include "TestSettingsManager.h"
#include "TestWavReader.h"
#include "TestWavWriter.h"

#include "TestFretboard.h"
#include "TestBrassInstrument.h"
#include "TestInputLips.h"
#include "TestMatlabParser.h"
#include "TestInstrumentParserBrass.h"
#include "TestScoreParserBrass.h"
#include "TestBreakpointFunction.h"

#include "TestGuitarString.h"
#include "TestInstrumentParserGuitar.h"
#include "TestScoreParserGuitar.h"

#include "TestMatlabFunction.h"
#include "TestMatlabFunction_StringDefGen.h"
#include "TestMatlabFunction_FretDefGen.h"
#include "TestMatlabFunction_ClusterGen.h"
#include "TestMatlabFunction_StrumGen.h"
#include "TestMatlabFunction_PluckGen.h"
#include "TestMatlabFunction_StrumGenMulti.h"

#include "TestBowedString.h"
#include "TestInstrumentParserBowedString.h"
#include "TestScoreParserBowedString.h"

#include "TestOutputModal.h"
#include "TestInputModalStrike.h"
#include "TestModalPlate.h"
#include "TestInstrumentParserModal.h"
#include "TestScoreParserModal.h"
#include "TestInputModalSine.h"

#include "TestConnectionNet1.h"

#ifdef USE_GPU
#include "TestGPUUtil.h"
#endif

#include <cppunit/ui/text/TestRunner.h>

#ifdef WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <cstdio>
using namespace std;

// required in order to link successfully to ModalPlate
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
    SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0, mydoc);
    sprintf(buf, "%s\\.ness", mydoc);
    mkdir(buf);
    strcat(buf, "\\cache\\");
    mkdir(buf);
#endif
}

int main(int argc, char *argv[])
{
    CppUnit::TextUi::TestRunner runner;

    runner.addTest(TestMaterial::suite());
    runner.addTest(TestMaterialsManager::suite());
    runner.addTest(TestGlobalSettings::suite());
    runner.addTest(TestLogger::suite());
    runner.addTest(TestCsrmatrix::suite());
    runner.addTest(TestBanded::suite());
    runner.addTest(TestPcg::suite());

    runner.addTest(TestAirbox::suite());
    runner.addTest(TestAirboxIndexed::suite());
    runner.addTest(TestBar::suite());
    runner.addTest(TestComponent::suite());
    runner.addTest(TestComponent1D::suite());
    runner.addTest(TestComponent2D::suite());
    runner.addTest(TestComponent3D::suite());
    runner.addTest(TestComponentString::suite());
    runner.addTest(TestPlate::suite());
    runner.addTest(TestPlateEmbedded::suite());
    runner.addTest(TestSoundBoard::suite());
    runner.addTest(TestStringWithFrets::suite());

    runner.addTest(TestInput::suite());
    runner.addTest(TestInputBow::suite());
    runner.addTest(TestInputPluck::suite());
    runner.addTest(TestInputSample::suite());
    runner.addTest(TestInputStrike::suite());
    runner.addTest(TestInputWav::suite());

    runner.addTest(TestOutput::suite());
    runner.addTest(TestOutputDifference::suite());
    runner.addTest(TestOutputPressure::suite());

    runner.addTest(TestConnectionP2P::suite());
    runner.addTest(TestConnectionZero::suite());
    runner.addTest(TestConnectionZeroPt1::suite());
    runner.addTest(TestEmbedding::suite());

    runner.addTest(TestParser::suite());    
    runner.addTest(TestInstrumentParserMP3D::suite());
    runner.addTest(TestInstrumentParserSoundboard::suite());
    runner.addTest(TestInstrumentParserXML::suite());
    runner.addTest(TestInstrumentParserZero::suite());
    runner.addTest(TestInstrumentParserZeroPt1::suite());
    runner.addTest(TestScoreParserSoundboard::suite());
    runner.addTest(TestScoreParserXML::suite());
    runner.addTest(TestScoreParserZero::suite());

    runner.addTest(TestInstrument::suite());
    runner.addTest(TestMathUtil::suite());
    runner.addTest(TestSettingsManager::suite());
    runner.addTest(TestWavReader::suite());
    runner.addTest(TestWavWriter::suite());

    runner.addTest(TestFretboard::suite());
    runner.addTest(TestBrassInstrument::suite());
    runner.addTest(TestInputLips::suite());
    runner.addTest(TestMatlabParser::suite());
    runner.addTest(TestInstrumentParserBrass::suite());
    runner.addTest(TestScoreParserBrass::suite());

    runner.addTest(TestBreakpointFunction::suite());

    runner.addTest(TestGuitarString::suite());
    runner.addTest(TestInstrumentParserGuitar::suite());
    runner.addTest(TestScoreParserGuitar::suite());

    runner.addTest(TestMatlabFunction::suite());
    runner.addTest(TestMatlabFunction_StringDefGen::suite());
    runner.addTest(TestMatlabFunction_FretDefGen::suite());
    runner.addTest(TestMatlabFunction_ClusterGen::suite());
    runner.addTest(TestMatlabFunction_StrumGen::suite());
    runner.addTest(TestMatlabFunction_PluckGen::suite());
    runner.addTest(TestMatlabFunction_StrumGenMulti::suite());

    runner.addTest(TestBowedString::suite());
    runner.addTest(TestInstrumentParserBowedString::suite());
    runner.addTest(TestScoreParserBowedString::suite());

    runner.addTest(TestOutputModal::suite());
    runner.addTest(TestInputModalStrike::suite());
    runner.addTest(TestModalPlate::suite());
    runner.addTest(TestInstrumentParserModal::suite());
    runner.addTest(TestScoreParserModal::suite());
    runner.addTest(TestInputModalSine::suite());

    runner.addTest(TestConnectionNet1::suite());

#ifdef USE_GPU
    runner.addTest(TestGPUUtil::suite());
#endif

    runner.run();
    return 0;
}
