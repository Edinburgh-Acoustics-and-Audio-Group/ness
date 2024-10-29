/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2013-2014. All rights reserved.
 */

#include "WavReader.h"
#include "Logger.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

// FIXME: this won't work on big endian systems

struct FmtHeader {
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
};

WavReader::WavReader(string filename)
{
    int i;
    char chunkID[4];
    int chunkSize;
    char format[4];
    bool foundData;
    bool foundHeader;
    FmtHeader hdr;

    logMessage(1, "Initialising WavReader for %s", filename.c_str());
    data = NULL;
    size = -1;
    sampleRate = -1;

    // load file
    ifstream f(filename.c_str(), ios::in | ios::binary);
    if (!f.good()) {
	logMessage(5, "Error opening file %s for reading", filename.c_str());
	return;
    }

    // read main chunk header
    f.read(chunkID, 4);
    if (strncmp(chunkID, "RIFF", 4)) {
	logMessage(5, "Unrecognised WAV file format in %s (no RIFF signature)", filename.c_str());
	f.close();
	return;
    }

    f.read((char *)&chunkSize, sizeof(int));
    f.read(format, 4);
    if (strncmp(format, "WAVE", 4)) {
	logMessage(5, "Unrecognised WAV file format in %s (no WAVE format)", filename.c_str());
	f.close();
	return;
    }
    
    // loop over sub-chunks
    foundData = false;
    foundHeader = false;
    while ((f.good()) && (!foundData)) {
	f.read(chunkID, 4);
	f.read((char *)&chunkSize, sizeof(int));

	if (!strncmp(chunkID, "fmt ", 4)) {
	    // found header
	    foundHeader = true;

	    if (chunkSize < sizeof(FmtHeader)) {
		logMessage(5, "Invalid WAV file format in %s (header too small)", filename.c_str());
		f.close();
		return;
	    }

	    // read the header
	    f.read((char *)&hdr, sizeof(FmtHeader));

	    if (hdr.audioFormat != 1) {
		logMessage(5, "Unrecognised audio format in %s - only PCM supported", filename.c_str());
		f.close();
		return;
	    }
	    
	    if (hdr.numChannels != 1) {
		logMessage(5, "Only mono WAV files supported for audio input - %s has %d channels",
			   filename.c_str(), hdr.numChannels);
		f.close();
		return;
	    }

	    sampleRate = hdr.sampleRate;

	    if (chunkSize > sizeof(FmtHeader)) {
		// skip any extra bytes
		f.ignore(chunkSize - sizeof(FmtHeader));
	    }
	}
	else if (!strncmp(chunkID, "data", 4)) {
	    // found data
	    foundData = true;

	    if (!foundHeader) {
		logMessage(5, "Invalid WAV file format in %s (no header before data)", filename.c_str());
		f.close();
		return;
	    }

	    if (hdr.bitsPerSample == 8) {
		size = chunkSize;

		unsigned char *buf = new unsigned char[size];
		f.read((char *)buf, size);
		data = new double[size];
		for (i = 0; i < size; i++) {
		    data[i] = ((double)buf[i]) - 128.0;
		}
		delete[] buf;
	    }
	    else if (hdr.bitsPerSample == 16) {
		size = chunkSize / 2;

		short *buf = new short[size];
		f.read((char *)buf, size*2);
		data = new double[size];
		for (i = 0; i < size; i++) {
		    data[i] = (double)buf[i];
		}
		delete[] buf;
	    }
	    else {
		logMessage(5, "Only 8 and 16-bit WAV files supported for audio input - %s is %d-bit",
			   filename.c_str(), hdr.bitsPerSample);
		f.close();
		return;
	    }
	    logMessage(1, "Read %d samples from %s at sample rate %d", size, filename.c_str(),
		       sampleRate);
	}
	else {
	    // skip past this one
	    f.ignore(chunkSize);
	}
    }

    if (!foundData) {
	logMessage(5, "Invalid WAV file format in %s - no data", filename.c_str());
    }

    f.close();
}

WavReader::~WavReader()
{
    delete[] data;
}

double *WavReader::getValues()
{
    return data;
}

int WavReader::getSize()
{
    return size;
}

int WavReader::getSampleRate()
{
    return sampleRate;
}

// normalise so maximum value is gain or -gain
void WavReader::normalise(double gain)
{
    int i;
    double max = 0.0;

    // find max value
    for (i = 0; i < size; i++) {
	if (fabs(data[i]) > max) max = fabs(data[i]);
    }

    if (max == 0.0) return;

    // normalise
    for (i = 0; i < size; i++) {
	data[i] = (data[i] / max) * gain;
    }
}

void WavReader::resampleTo(int sr)
{
    if (sampleRate == sr) return;

    logMessage(1, "Resampling WAV data from %d to %dHz", sampleRate, sr);

    // work out new size first
    int newSize = (int)(((double)size*(double)sr) / ((double)sampleRate));
    double *newData = new double[newSize];

    double pos = 0.0;
    double incr = ((double)size) / ((double)newSize);

    // actually resample
    for (int i = 0; i < newSize; i++) {
	// work out where we are in the old data
	int idx = (int)pos;
	double alpha = pos - ((double)idx);

	if (idx >= (size-1)) {
	    newData[i] = data[size - 1];
	}
	else {
	    newData[i] = ((1.0 - alpha) * data[idx]) + (alpha * data[idx+1]);
	}

	pos += incr;
    }

    // replace the old data
    delete[] data;
    data = newData;
    size = newSize;
    sampleRate = sr;
}

