#ifndef INGESTION_H
#define INGESTION_H

#include <string>
#include "types.h"

// Sequential baseline read
bool loadBinarySequential(const string& filePath, ComplexFloat* buffer, 
                          int lines, int totalLineElements, int samplesPerEcho, 
                          int validSamples, int numPols, int polIndex);

// OpenMP multi-threaded parallel read
bool loadBinaryParallel(const string& filePath, ComplexFloat* buffer, 
                        int lines, int totalLineElements, int samplesPerEcho, 
                        int validSamples, int numPols, int polIndex, int numThreads);

// Helper to generate a dummy binary file for testing if it doesn't exist
bool generateDummyBinary(const std::string& filePath, int lines, int totalLineElements);

#endif // INGESTION_H
