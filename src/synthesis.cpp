#include "synthesis.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <algorithm>

bool synthesizeImage(const ComplexFloat* h_data, int lines, int samples, const std::string& outputPath, double& synthesisSecs) {
    double tStart = omp_get_wtime();
    size_t numElements = static_cast<size_t>(lines) * samples;

    std::cout << "[Phase 4] Starting magnitude calculation and 2.5-sigma contrast stretching..." << std::endl;

    std::vector<float> magnitudes(numElements);
    double sum = 0.0;

    // Task 4.2: Magnitude Calculation and sum accumulation using OpenMP
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < numElements; ++i) {
        float realVal = h_data[i].r;
        float imagVal = h_data[i].i;
        // Absolute magnitude scaled by lines for unnormalized inverse cuFFT
        float mag = std::sqrt(realVal * realVal + imagVal * imagVal) / lines;
        magnitudes[i] = mag;
        sum += mag;
    }

    float mean = static_cast<float>(sum / numElements);

    // Compute standard deviation
    double sq_sum = 0.0;
    #pragma omp parallel for reduction(+:sq_sum)
    for (size_t i = 0; i < numElements; ++i) {
        float diff = magnitudes[i] - mean;
        sq_sum += diff * diff;
    }
    float stddev = std::sqrt(static_cast<float>(sq_sum / numElements));

    // Apply 2.5-sigma clipping (standard practice for high-dynamic-range radar imagery)
    // This removes bright noise spikes/calibration frame outliers and stretches active pixels
    float minClip = mean - 2.5f * stddev;
    float maxClip = mean + 2.5f * stddev;
    
    if (minClip < 0.0f) minClip = 0.0f;
    if (maxClip <= minClip) maxClip = minClip + 1e-5f; // Prevent divide by zero

    float range = maxClip - minClip;

    std::cout << "  - Image Stats: Mean = " << mean << ", StdDev = " << stddev << std::endl;
    std::cout << "  - Scaling Range (2.5-sigma): [" << minClip << ", " << maxClip << "]" << std::endl;

    // Allocate 1 byte per pixel for grayscale PGM
    std::vector<unsigned char> pixels(numElements);

    // Map intensities to 0-255 scale using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < numElements; ++i) {
        float val = magnitudes[i];
        if (val > maxClip) val = maxClip;
        if (val < minClip) val = minClip;
        
        float normalized = (val - minClip) / range;
        pixels[i] = static_cast<unsigned char>(normalized * 255.0f);
    }

    // Task 4.3: Save as PGM (P5 format)
    std::cout << "  - Writing image to PGM file: " << outputPath << "..." << std::endl;
    std::ofstream file(outputPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output image file: " << outputPath << std::endl;
        return false;
    }

    // PGM P5 header
    file << "P5\n" << samples << " " << lines << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), numElements);
    file.close();

    double tEnd = omp_get_wtime();
    synthesisSecs = tEnd - tStart;
    
    std::cout << "Image synthesis completed in " << synthesisSecs << " seconds." << std::endl;
    return true;
}
