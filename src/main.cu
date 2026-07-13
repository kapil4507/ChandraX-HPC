#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include "parser.h"
#include "ingestion.h"
#include "synthesis.h"
#include "types.h"

// Conditionally include GPU processing header only if compiled with a CUDA compiler
#ifdef __CUDACC__
#include "processing.cuh"

using namespace std;
#endif

int main(int argc, char* argv[]) {
    string xmlPath = "data/sample_label.xml";
    int polIndex = 0; // 0 for LH, 1 for LV

    if (argc > 1) {
        xmlPath = argv[1];
    }
    if (argc > 2) {
        polIndex = stoi(argv[2]);
    }

    cout << "==================================================" << endl;
    cout << "DFSAR Synthesis & Feature Extraction Pipeline" << endl;
    cout << "Phase 1: Environment Setup & Data Parsing" << endl;
    cout << "Phase 2: High-Throughput Data Ingestion" << endl;
    cout << "Phase 3: GPU Memory & Mathematical Transformation" << endl;
    cout << "Phase 4: Image Synthesis & Benchmarking" << endl;
    cout << "==================================================" << endl;

    // --- PHASE 1: Data Parsing ---
    cout << "\n[Phase 1] Parsing XML label file: " << xmlPath << "..." << endl;
    MatrixDimensions dims = parseXML(xmlPath);

    if (dims.lines == 0 || dims.totalLineElements == 0 || dims.samplesPerEcho == 0) {
        cerr << "Error: Failed to parse dimensions or parameters from XML." << endl;
        return 1;
    }

    cout << "Successfully parsed parameters from XML:" << endl;
    cout << "  - Lines (Rows):           " << dims.lines << endl;
    cout << "  - Total Line Elements:    " << dims.totalLineElements << " bytes/line" << endl;
    cout << "  - Samples per Echo:       " << dims.samplesPerEcho << " complex samples/line" << endl;
    cout << "  - Valid Cropped Samples:  " << dims.validSamples << " complex samples/line" << endl;
    cout << "  - Number of Polarizations:" << dims.numPols << endl;
    cout << "  - Data Type:              " << dims.dataType << endl;
    cout << "  - Target Binary File:     " << dims.fileName << endl;

    if (polIndex < 0 || polIndex >= dims.numPols) {
        cerr << "Error: Selected polarization index " << polIndex 
                  << " is invalid. Must be between 0 and " << (dims.numPols - 1) << endl;
        return 1;
    }
    cout << "Selected Polarization Index: " << polIndex 
              << " (" << (polIndex == 0 ? "LH" : (polIndex == 1 ? "LV" : "Unknown")) << ")" << endl;

    // Calculate memory size for host allocation (lines * validSamples complex floats)
    size_t numElements = static_cast<size_t>(dims.lines) * dims.validSamples;
    size_t numBytes = numElements * sizeof(ComplexFloat);
    double numMB = static_cast<double>(numBytes) / (1024.0 * 1024.0);

    // Calculate file size on disk
    size_t expectedFileSizeBytes = static_cast<size_t>(dims.lines) * dims.totalLineElements;
    double fileMB = static_cast<double>(expectedFileSizeBytes) / (1024.0 * 1024.0);

    cout << "\nFile & Memory Configurations:" << endl;
    cout << "  - Expected File Size: " << expectedFileSizeBytes << " bytes (" << fileMB << " MB)" << endl;
    cout << "  - Processed Elements: " << numElements << " (ComplexFloat)" << endl;
    cout << "  - Host RAM Required:  " << numBytes << " bytes (" << numMB << " MB)" << endl;

    // Dynamically allocate memory array to hold the extracted complex data
    cout << "Allocating memory on host..." << endl;
    ComplexFloat* dataArray = nullptr;
    try {
        dataArray = new ComplexFloat[numElements];
        cout << "Memory allocation successful! Address: " << dataArray << endl;
    } catch (const bad_alloc& e) {
        cerr << "Memory allocation failed: " << e.what() << endl;
        return 1;
    }

    // Set up binary path
    string binPath = "data/" + dims.fileName;

    // Check if binary file exists; generate a synthetic one if not
    ifstream checkFile(binPath, ios::binary | ios::ate);
    if (!checkFile.is_open()) {
        cout << "\nBinary file not found at " << binPath << ". Generating dummy binary data..." << endl;
        if (!generateDummyBinary(binPath, dims.lines, dims.totalLineElements)) {
            cerr << "Error: Failed to generate dummy binary file." << endl;
            delete[] dataArray;
            return 1;
        }
        cout << "Dummy binary file generated successfully." << endl;
    } else {
        size_t fileSize = checkFile.tellg();
        checkFile.close();
        cout << "\nBinary file found at " << binPath << ". Size: " << fileSize 
                  << " bytes (Expected: " << expectedFileSizeBytes << " bytes)" << endl;
    }

    // --- PHASE 2: Ingestion ---
    cout << "\n[Phase 2] Benchmarking Sequential Ingestion..." << endl;
    double tStartSeq = omp_get_wtime();
    bool seqSuccess = loadBinarySequential(binPath, dataArray, dims.lines, 
                                           dims.totalLineElements, dims.samplesPerEcho, 
                                           dims.validSamples, dims.numPols, polIndex);
    double tEndSeq = omp_get_wtime();
    double seqTime = tEndSeq - tStartSeq;

    if (seqSuccess) {
        cout << "Sequential ingestion completed in " << seqTime << " seconds." << endl;
        cout << "Ingestion throughput (processed complex data): " << numMB / seqTime << " MB/s" << endl;
        cout << "I/O read throughput (raw file load): " << fileMB / seqTime << " MB/s" << endl;
    } else {
        cerr << "Sequential ingestion failed." << endl;
        delete[] dataArray;
        return 1;
    }

    // Benchmark Parallel Ingestion with different thread counts
    int threadsList[] = {2, 4, 8, 16};
    double bestParTime = seqTime;
    int bestThreadCount = 1;

    for (int t : threadsList) {
        cout << "\nBenchmarking Parallel Ingestion with " << t << " threads..." << endl;
        
        // Zero out memory to ensure no caching artifacts
        fill(dataArray, dataArray + numElements, ComplexFloat{0.0f, 0.0f});

        double tStartPar = omp_get_wtime();
        bool parSuccess = loadBinaryParallel(binPath, dataArray, dims.lines, 
                                             dims.totalLineElements, dims.samplesPerEcho, 
                                             dims.validSamples, dims.numPols, polIndex, t);
        double tEndPar = omp_get_wtime();
        double parTime = tEndPar - tStartPar;

        if (parSuccess) {
            cout << "Parallel ingestion (" << t << " threads) completed in " << parTime << " seconds." << endl;
            cout << "Throughput: " << fileMB / parTime << " MB/s (raw file read)" << endl;
            cout << "Speedup: " << seqTime / parTime << "x" << endl;

            if (parTime < bestParTime) {
                bestParTime = parTime;
                bestThreadCount = t;
            }
        } else {
            cerr << "Parallel ingestion with " << t << " threads failed." << endl;
        }
    }

    // Load the final data using the best thread configuration
    cout << "\nLoading final data using the best parallel thread configuration (" << bestThreadCount << " threads)..." << endl;
    loadBinaryParallel(binPath, dataArray, dims.lines, dims.totalLineElements, 
                       dims.samplesPerEcho, dims.validSamples, dims.numPols, polIndex, bestThreadCount);

    // --- PHASE 3: GPU Processing ---
    double gpuTime = 0.0;
#ifdef __CUDACC__
    cout << "\n[Phase 3] Running GPU Processing Pipeline (Range-Doppler Algorithm)..." << endl;
    bool gpuSuccess = runGPUProcessing(dataArray, dims.lines, dims.validSamples, 
                                       dims.centerFrequency, dims.slantRange, dims.prf,
                                       gpuTime);
    if (!gpuSuccess) {
        cerr << "GPU processing failed." << endl;
        delete[] dataArray;
        return 1;
    }
#else
    cout << "\n[Phase 3] CUDA Compiler not active (CPU compilation). Skipping GPU acceleration step." << endl;
    cout << "          (To compile with CUDA on Ramanujan Cluster, run build.sh with nvcc available)." << endl;
#endif

    // --- PHASE 4: Image Synthesis & Benchmarking ---
    double synthesisTime = 0.0;
    string outImagePath = "data/dfsar_focused_lunar_surface.pgm";
    bool synthSuccess = synthesizeImage(dataArray, dims.lines, dims.validSamples, outImagePath, synthesisTime);
    
    if (!synthSuccess) {
        cerr << "Image synthesis failed." << endl;
        delete[] dataArray;
        return 1;
    }

    // Performance Benchmarking Summary
    cout << "\n==================================================" << endl;
    cout << "PERFORMANCE PROFILE REPORT" << endl;
    cout << "==================================================" << endl;
    cout << "Raw Data File Size:           " << fileMB << " MB" << endl;
    cout << "Processed RAM Matrix Size:    " << numMB << " MB" << endl;
    cout << "Sequential Ingestion Time:     " << seqTime << " s (" << (fileMB / seqTime) << " MB/s)" << endl;
    cout << "OpenMP Ingestion Time (" << bestThreadCount << " th):   " << bestParTime << " s (" << (fileMB / bestParTime) << " MB/s)" << endl;
    cout << "OpenMP Speedup Factor:        " << (seqTime / bestParTime) << "x" << endl;
#ifdef __CUDACC__
    cout << "GPU Transform & FFT Time:      " << gpuTime << " s" << endl;
#else
    cout << "GPU Transform & FFT Time:      N/A (Skipped)" << endl;
#endif
    cout << "Image Synthesis Time (CPU):    " << synthesisTime << " s" << endl;
    
    double totalTime = bestParTime + gpuTime + synthesisTime;
    cout << "--------------------------------------------------" << endl;
    cout << "Total Pipeline Processing Time: " << totalTime << " s" << endl;
    cout << "==================================================" << endl;

    // Clean up
    cout << "\nDeallocating matrix..." << endl;
    delete[] dataArray;
    cout << "Cleanup completed. Pipeline executed successfully." << endl;
    cout << "==================================================" << endl;

    return 0;
}
