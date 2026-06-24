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
#endif

int main(int argc, char* argv[]) {
    std::string xmlPath = "data/sample_label.xml";
    int polIndex = 0; // 0 for LH, 1 for LV

    if (argc > 1) {
        xmlPath = argv[1];
    }
    if (argc > 2) {
        polIndex = std::stoi(argv[2]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "DFSAR Synthesis & Feature Extraction Pipeline" << std::endl;
    std::cout << "Phase 1: Environment Setup & Data Parsing" << std::endl;
    std::cout << "Phase 2: High-Throughput Data Ingestion" << std::endl;
    std::cout << "Phase 3: GPU Memory & Mathematical Transformation" << std::endl;
    std::cout << "Phase 4: Image Synthesis & Benchmarking" << std::endl;
    std::cout << "==================================================" << std::endl;

    // --- PHASE 1: Data Parsing ---
    std::cout << "\n[Phase 1] Parsing XML label file: " << xmlPath << "..." << std::endl;
    MatrixDimensions dims = parseXML(xmlPath);

    if (dims.lines == 0 || dims.totalLineElements == 0 || dims.samplesPerEcho == 0) {
        std::cerr << "Error: Failed to parse dimensions or parameters from XML." << std::endl;
        return 1;
    }

    std::cout << "Successfully parsed parameters from XML:" << std::endl;
    std::cout << "  - Lines (Rows):           " << dims.lines << std::endl;
    std::cout << "  - Total Line Elements:    " << dims.totalLineElements << " bytes/line" << std::endl;
    std::cout << "  - Samples per Echo:       " << dims.samplesPerEcho << " complex samples/line" << std::endl;
    std::cout << "  - Number of Polarizations:" << dims.numPols << std::endl;
    std::cout << "  - Data Type:              " << dims.dataType << std::endl;
    std::cout << "  - Target Binary File:     " << dims.fileName << std::endl;

    if (polIndex < 0 || polIndex >= dims.numPols) {
        std::cerr << "Error: Selected polarization index " << polIndex 
                  << " is invalid. Must be between 0 and " << (dims.numPols - 1) << std::endl;
        return 1;
    }
    std::cout << "Selected Polarization Index: " << polIndex 
              << " (" << (polIndex == 0 ? "LH" : (polIndex == 1 ? "LV" : "Unknown")) << ")" << std::endl;

    // Calculate memory size for host allocation (lines * samplesPerEcho complex floats)
    size_t numElements = static_cast<size_t>(dims.lines) * dims.samplesPerEcho;
    size_t numBytes = numElements * sizeof(ComplexFloat);
    double numMB = static_cast<double>(numBytes) / (1024.0 * 1024.0);

    // Calculate file size on disk
    size_t expectedFileSizeBytes = static_cast<size_t>(dims.lines) * dims.totalLineElements;
    double fileMB = static_cast<double>(expectedFileSizeBytes) / (1024.0 * 1024.0);

    std::cout << "\nFile & Memory Configurations:" << std::endl;
    std::cout << "  - Expected File Size: " << expectedFileSizeBytes << " bytes (" << fileMB << " MB)" << std::endl;
    std::cout << "  - Processed Elements: " << numElements << " (ComplexFloat)" << std::endl;
    std::cout << "  - Host RAM Required:  " << numBytes << " bytes (" << numMB << " MB)" << std::endl;

    // Dynamically allocate memory array to hold the extracted complex data
    std::cout << "Allocating memory on host..." << std::endl;
    ComplexFloat* dataArray = nullptr;
    try {
        dataArray = new ComplexFloat[numElements];
        std::cout << "Memory allocation successful! Address: " << dataArray << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return 1;
    }

    // Set up binary path
    std::string binPath = "data/" + dims.fileName;

    // Check if binary file exists; generate a synthetic one if not
    std::ifstream checkFile(binPath, std::ios::binary | std::ios::ate);
    if (!checkFile.is_open()) {
        std::cout << "\nBinary file not found at " << binPath << ". Generating dummy binary data..." << std::endl;
        if (!generateDummyBinary(binPath, dims.lines, dims.totalLineElements)) {
            std::cerr << "Error: Failed to generate dummy binary file." << std::endl;
            delete[] dataArray;
            return 1;
        }
        std::cout << "Dummy binary file generated successfully." << std::endl;
    } else {
        size_t fileSize = checkFile.tellg();
        checkFile.close();
        std::cout << "\nBinary file found at " << binPath << ". Size: " << fileSize 
                  << " bytes (Expected: " << expectedFileSizeBytes << " bytes)" << std::endl;
    }

    // --- PHASE 2: Ingestion ---
    std::cout << "\n[Phase 2] Benchmarking Sequential Ingestion..." << std::endl;
    double tStartSeq = omp_get_wtime();
    bool seqSuccess = loadBinarySequential(binPath, dataArray, dims.lines, 
                                           dims.totalLineElements, dims.samplesPerEcho, 
                                           dims.numPols, polIndex);
    double tEndSeq = omp_get_wtime();
    double seqTime = tEndSeq - tStartSeq;

    if (seqSuccess) {
        std::cout << "Sequential ingestion completed in " << seqTime << " seconds." << std::endl;
        std::cout << "Ingestion throughput (processed complex data): " << numMB / seqTime << " MB/s" << std::endl;
        std::cout << "I/O read throughput (raw file load): " << fileMB / seqTime << " MB/s" << std::endl;
    } else {
        std::cerr << "Sequential ingestion failed." << std::endl;
        delete[] dataArray;
        return 1;
    }

    // Benchmark Parallel Ingestion with different thread counts
    int threadsList[] = {2, 4, 8, 16};
    double bestParTime = seqTime;
    int bestThreadCount = 1;

    for (int t : threadsList) {
        std::cout << "\nBenchmarking Parallel Ingestion with " << t << " threads..." << std::endl;
        
        // Zero out memory to ensure no caching artifacts
        std::fill(dataArray, dataArray + numElements, ComplexFloat{0.0f, 0.0f});

        double tStartPar = omp_get_wtime();
        bool parSuccess = loadBinaryParallel(binPath, dataArray, dims.lines, 
                                             dims.totalLineElements, dims.samplesPerEcho, 
                                             dims.numPols, polIndex, t);
        double tEndPar = omp_get_wtime();
        double parTime = tEndPar - tStartPar;

        if (parSuccess) {
            std::cout << "Parallel ingestion (" << t << " threads) completed in " << parTime << " seconds." << std::endl;
            std::cout << "Throughput: " << fileMB / parTime << " MB/s (raw file read)" << std::endl;
            std::cout << "Speedup: " << seqTime / parTime << "x" << std::endl;

            if (parTime < bestParTime) {
                bestParTime = parTime;
                bestThreadCount = t;
            }
        } else {
            std::cerr << "Parallel ingestion with " << t << " threads failed." << std::endl;
        }
    }

    // Load the final data using the best thread configuration
    std::cout << "\nLoading final data using the best parallel thread configuration (" << bestThreadCount << " threads)..." << std::endl;
    loadBinaryParallel(binPath, dataArray, dims.lines, dims.totalLineElements, 
                       dims.samplesPerEcho, dims.numPols, polIndex, bestThreadCount);

    // --- PHASE 3: GPU Processing ---
    double gpuTime = 0.0;
#ifdef __CUDACC__
    std::cout << "\n[Phase 3] Running GPU Processing Pipeline (Range-Doppler Algorithm)..." << std::endl;
    bool gpuSuccess = runGPUProcessing(dataArray, dims.lines, dims.samplesPerEcho, gpuTime);
    if (!gpuSuccess) {
        std::cerr << "GPU processing failed." << std::endl;
        delete[] dataArray;
        return 1;
    }
#else
    std::cout << "\n[Phase 3] CUDA Compiler not active (CPU compilation). Skipping GPU acceleration step." << std::endl;
    std::cout << "          (To compile with CUDA on Ramanujan Cluster, run build.sh with nvcc available)." << std::endl;
#endif

    // --- PHASE 4: Image Synthesis & Benchmarking ---
    double synthesisTime = 0.0;
    std::string outImagePath = "data/dfsar_focused_lunar_surface.pgm";
    bool synthSuccess = synthesizeImage(dataArray, dims.lines, dims.samplesPerEcho, outImagePath, synthesisTime);
    
    if (!synthSuccess) {
        std::cerr << "Image synthesis failed." << std::endl;
        delete[] dataArray;
        return 1;
    }

    // Performance Benchmarking Summary
    std::cout << "\n==================================================" << std::endl;
    std::cout << "PERFORMANCE PROFILE REPORT" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Raw Data File Size:           " << fileMB << " MB" << std::endl;
    std::cout << "Processed RAM Matrix Size:    " << numMB << " MB" << std::endl;
    std::cout << "Sequential Ingestion Time:     " << seqTime << " s (" << (fileMB / seqTime) << " MB/s)" << std::endl;
    std::cout << "OpenMP Ingestion Time (" << bestThreadCount << " th):   " << bestParTime << " s (" << (fileMB / bestParTime) << " MB/s)" << std::endl;
    std::cout << "OpenMP Speedup Factor:        " << (seqTime / bestParTime) << "x" << std::endl;
#ifdef __CUDACC__
    std::cout << "GPU Transform & FFT Time:      " << gpuTime << " s" << std::endl;
#else
    std::cout << "GPU Transform & FFT Time:      N/A (Skipped)" << std::endl;
#endif
    std::cout << "Image Synthesis Time (CPU):    " << synthesisTime << " s" << std::endl;
    
    double totalTime = bestParTime + gpuTime + synthesisTime;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Total Pipeline Processing Time: " << totalTime << " s" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Clean up
    std::cout << "\nDeallocating matrix..." << std::endl;
    delete[] dataArray;
    std::cout << "Cleanup completed. Pipeline executed successfully." << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
