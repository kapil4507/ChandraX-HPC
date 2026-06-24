#include "ingestion.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

bool loadBinarySequential(const std::string& filePath, ComplexFloat* buffer, 
                          int lines, int totalLineElements, int samplesPerEcho, 
                          int numPols, int polIndex) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for sequential read: " << filePath << std::endl;
        return false;
    }

    int header_bytes = totalLineElements - numPols * samplesPerEcho * 2;
    int polOffset = header_bytes + polIndex * samplesPerEcho * 2;
    
    // Process in batches of lines to optimize I/O overhead
    const int batchLines = 4096;
    size_t batchBytes = static_cast<size_t>(batchLines) * totalLineElements;
    std::vector<char> rawBatch(batchBytes);

    int linesRead = 0;
    while (linesRead < lines) {
        int currentBatchLines = std::min(batchLines, lines - linesRead);
        size_t currentBatchBytes = static_cast<size_t>(currentBatchLines) * totalLineElements;
        
        file.read(rawBatch.data(), currentBatchBytes);
        std::streamsize bytesRead = file.gcount();
        if (bytesRead < static_cast<std::streamsize>(currentBatchBytes)) {
            // Adjust batch lines based on actual read bytes
            currentBatchLines = static_cast<int>(bytesRead / totalLineElements);
            if (currentBatchLines == 0) break;
        }

        // Process bytes from batch in memory
        for (int line = 0; line < currentBatchLines; ++line) {
            size_t lineOffset = static_cast<size_t>(line) * totalLineElements;
            const char* polData = rawBatch.data() + lineOffset + polOffset;
            
            for (int j = 0; j < samplesPerEcho; ++j) {
                int8_t I = static_cast<int8_t>(polData[2 * j]);
                int8_t Q = static_cast<int8_t>(polData[2 * j + 1]);
                
                size_t outIdx = static_cast<size_t>(linesRead + line) * samplesPerEcho + j;
                buffer[outIdx].r = static_cast<float>(I);
                buffer[outIdx].i = static_cast<float>(Q);
            }
        }
        linesRead += currentBatchLines;
    }

    file.close();
    return true;
}

bool loadBinaryParallel(const std::string& filePath, ComplexFloat* buffer, 
                        int lines, int totalLineElements, int samplesPerEcho, 
                        int numPols, int polIndex, int numThreads) {
    bool success = true;
    int header_bytes = totalLineElements - numPols * samplesPerEcho * 2;
    int polOffset = header_bytes + polIndex * samplesPerEcho * 2;

    omp_set_num_threads(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int actualThreads = omp_get_num_threads();

        int chunkLines = lines / actualThreads;
        int remainder = lines % actualThreads;

        int startLine = tid * chunkLines;
        int linesToRead = chunkLines;

        // The last thread handles the remainder
        if (tid == actualThreads - 1) {
            linesToRead += remainder;
        }

        if (linesToRead > 0) {
            std::ifstream threadFile(filePath, std::ios::binary);
            if (!threadFile.is_open()) {
                #pragma omp critical
                {
                    std::cerr << "Error: Thread " << tid << " could not open file: " << filePath << std::endl;
                    success = false;
                }
            } else {
                // Seek to start position of this thread's chunk
                size_t startOffset = static_cast<size_t>(startLine) * totalLineElements;
                threadFile.seekg(startOffset);

                // Allocate buffer to read the entire chunk in a single parallel read operation
                size_t bytesToRead = static_cast<size_t>(linesToRead) * totalLineElements;
                std::vector<char> rawChunk(bytesToRead);

                threadFile.read(rawChunk.data(), bytesToRead);
                std::streamsize bytesRead = threadFile.gcount();
                if (bytesRead < static_cast<std::streamsize>(bytesToRead)) {
                    // Adjust lines to process if file was smaller than expected
                    linesToRead = static_cast<int>(bytesRead / totalLineElements);
                }
                threadFile.close();

                // Process the read bytes in memory
                for (int line = 0; line < linesToRead; ++line) {
                    size_t lineOffset = static_cast<size_t>(line) * totalLineElements;
                    const char* polData = rawChunk.data() + lineOffset + polOffset;

                    for (int j = 0; j < samplesPerEcho; ++j) {
                        int8_t I = static_cast<int8_t>(polData[2 * j]);
                        int8_t Q = static_cast<int8_t>(polData[2 * j + 1]);
                        
                        size_t outIdx = static_cast<size_t>(startLine + line) * samplesPerEcho + j;
                        buffer[outIdx].r = static_cast<float>(I);
                        buffer[outIdx].i = static_cast<float>(Q);
                    }
                }
            }
        }
    }

    return success;
}

bool generateDummyBinary(const std::string& filePath, int lines, int totalLineElements) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing dummy data: " << filePath << std::endl;
        return false;
    }

    // Allocate single line buffer to avoid large RAM footprint during creation
    std::vector<char> dummyLine(totalLineElements, 0);

    // Default structural parameters for raw DFSAR data mapping
    int samplesPerEcho = 1024;
    int numPols = 2;
    int header_bytes = totalLineElements - numPols * samplesPerEcho * 2;

    for (int line = 0; line < lines; ++line) {
        // Write mock synchronization pattern in the header
        if (header_bytes >= 2) {
            dummyLine[0] = 0x7F;
            dummyLine[1] = 0x7F;
        }
        
        // Populate LH and LV polarizations with a synthetic frequency-shifting chirp
        for (int pol = 0; pol < numPols; ++pol) {
            int polOffset = header_bytes + pol * samplesPerEcho * 2;
            for (int j = 0; j < samplesPerEcho; ++j) {
                // Synthetic phase chirp computation matching integer scaling (-128 to 127)
                float phase = static_cast<float>(line) * 0.005f + static_cast<float>(j) * 0.02f;
                float I_val = std::cos(phase) * 60.0f;
                float Q_val = std::sin(phase) * 60.0f;

                dummyLine[polOffset + 2 * j] = static_cast<char>(static_cast<int8_t>(I_val));
                dummyLine[polOffset + 2 * j + 1] = static_cast<char>(static_cast<int8_t>(Q_val));
            }
        }

        file.write(dummyLine.data(), totalLineElements);
    }

    file.close();
    return true;
}
