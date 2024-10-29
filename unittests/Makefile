CXX=g++
CC=gcc
COMMONDIR=../src/lib
CFLAGS=-I$(COMMONDIR) -Wno-write-strings -O3 -DUSE_SSE -DUSE_AVX `xml2-config --cflags` -I../src
LDFLAGS=-lpthread `xml2-config --libs` -lcppunit

TEST_OBJECTS=main.o DummyAirbox.o DummyComponent1D.o DummyComponent2D.o DummyComponent3D.o DummyConnectionP2P.o DummyInputSample.o DummyParser.o TestBanded.o TestBreakpointFunction.o TestCsrmatrix.o TestGlobalSettings.o TestLogger.o TestMaterial.o TestMaterialsManager.o TestPcg.o TestAirbox.o TestAirboxIndexed.o TestBar.o TestComponent.o TestComponent1D.o TestComponent2D.o TestComponent3D.o TestComponentString.o TestPlate.o TestPlateEmbedded.o TestSoundBoard.o TestStringWithFrets.o TestInput.o TestInputBow.o TestInputPluck.o TestInputSample.o TestInputStrike.o TestInputWav.o TestOutput.o TestOutputDifference.o TestOutputPressure.o TestConnectionP2P.o TestConnectionZero.o TestConnectionZeroPt1.o TestEmbedding.o TestInstrumentParserMP3D.o TestInstrumentParserSoundboard.o TestInstrumentParserXML.o TestInstrumentParserZero.o TestInstrumentParserZeroPt1.o TestParser.o TestScoreParserSoundboard.o TestScoreParserXML.o TestScoreParserZero.o TestInstrument.o TestMathUtil.o TestSettingsManager.o TestWavReader.o TestWavWriter.o TestFretboard.o TestBrassInstrument.o TestInputLips.o TestMatlabParser.o TestInstrumentParserBrass.o TestScoreParserBrass.o TestGuitarString.o TestInstrumentParserGuitar.o TestScoreParserGuitar.o TestMatlabFunction.o TestMatlabFunction_StringDefGen.o TestMatlabFunction_FretDefGen.o TestMatlabFunction_ClusterGen.o TestMatlabFunction_StrumGen.o TestMatlabFunction_PluckGen.o TestMatlabFunction_StrumGenMulti.o TestBowedString.o TestInstrumentParserBowedString.o TestScoreParserBowedString.o TestOutputModal.o TestInputModalStrike.o TestModalPlate.o TestInstrumentParserModal.o TestConnectionNet1.o TestInputModalSine.o TestScoreParserModal.o

FRAMEWORK_OBJECTS=../src/Airbox.o ../src/AirboxIndexed.o ../src/Bar.o ../src/BowedString.o ../src/BrassInstrument.o ../src/BrassInstrument_avx.o ../src/BreakpointFunction.o ../src/Component.o ../src/Component1D.o ../src/Component2D.o ../src/Component3D.o ../src/ComponentString.o ../src/Connection.o ../src/ConnectionNet1.o ../src/ConnectionP2P.o ../src/ConnectionZero.o ../src/ConnectionZeroPt1.o ../src/Embedding.o ../src/Fretboard.o ../src/GlobalSettings.o ../src/GuitarString.o ../src/Input.o ../src/InputBow.o ../src/InputLips.o ../src/InputModalSine.o ../src/InputModalStrike.o ../src/InputPluck.o ../src/InputSample.o ../src/InputStrike.o ../src/InputValve.o ../src/InputWav.o ../src/Instrument.o ../src/InstrumentParser.o ../src/InstrumentParserBowedString.o ../src/InstrumentParserBrass.o ../src/InstrumentParserGuitar.o ../src/InstrumentParserModal.o ../src/InstrumentParserMP3D.o ../src/InstrumentParserSoundboard.o ../src/InstrumentParserXML.o ../src/InstrumentParserZero.o ../src/InstrumentParserZeroPt1.o ../src/Logger.o ../src/Material.o ../src/MaterialsManager.o ../src/MatlabFunction.o ../src/MatlabFunction_ClusterGen.o ../src/MatlabFunction_FretDefGen.o ../src/MatlabFunction_PluckGen.o ../src/MatlabFunction_StringDefGen.o ../src/MatlabFunction_StrumGen.o ../src/MatlabFunction_StrumGenMulti.o ../src/MatlabParser.o ../src/MathUtil.o ../src/ModalPlate.o ../src/ModalPlate_avx.o ../src/Output.o ../src/OutputDifference.o ../src/OutputModal.o ../src/OutputPressure.o ../src/Parser.o ../src/Plate.o ../src/PlateEmbedded.o ../src/Profiler.o ../src/ScoreParser.o ../src/ScoreParserBowedString.o ../src/ScoreParserBrass.o ../src/ScoreParserGuitar.o ../src/ScoreParserModal.o ../src/ScoreParserMP3D.o ../src/ScoreParserSoundboard.o ../src/ScoreParserXML.o ../src/ScoreParserZero.o ../src/ScoreParserZeroPt1.o ../src/SettingsManager.o ../src/SoundBoard.o ../src/StringWithFrets.o ../src/Task.o ../src/TaskModalPlateEnd.o ../src/TaskModalPlateRow.o ../src/TaskWholeComponent.o ../src/TaskWholeConnection.o ../src/WavReader.o ../src/WavWriter.o ../src/WorkerThread.o ../src/matgen.o

COMMON_OBJECTS=$(COMMONDIR)/csrmatrix.o $(COMMONDIR)/arralloc.o $(COMMONDIR)/pcg.o $(COMMONDIR)/sse.o $(COMMONDIR)/banded.o

OBJECTS=$(FRAMEWORK_OBJECTS) $(TEST_OBJECTS) $(COMMON_OBJECTS)

all: main

main: $(OBJECTS)
	$(CXX) -o main $(OBJECTS) $(LDFLAGS)

.cpp.o:
	$(CXX) -c $< -o $@ $(CFLAGS)

.c.o:
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f $(TEST_OBJECTS) main
