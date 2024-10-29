/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * Writes Wav files to disk (both mono single channel files and stereo mixdown)
 */
#ifndef _WAVWRITER_H_
#define _WAVWRITER_H_

#include "Output.h"

#include <string>
#include <vector>
using namespace std;

// WAV file header structure
struct WavHeader {
    char chunkID[4];
    int chunkSize;
    char format[4];

    char subChunkID[4];
    int subChunkSize;
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;

    char subChunkID2[4];
    int subChunkSize2;
};

class WavWriter {
 public:
    WavWriter(string filename);
    ~WavWriter();

    bool writeMonoWavFile(Output *output, double max);
    bool writeStereoMix(vector<Output*> *outputs, double max);

 private:
    string filename;

    bool writeMonoWav(short *data, int len, int sr);

    // len is length in samples. data is interleaved, left then right
    bool writeStereoWav(short *data, int len, int sr);

    void initWavHeader(WavHeader *hdr, bool stereo, int sampleRate, int len);
};

#endif
