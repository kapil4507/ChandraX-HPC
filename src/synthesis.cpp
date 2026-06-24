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

    std::cout << "[Phase 4] Starting magnitude calculation and image mapping..." << std::endl;

    std::vector<float> magnitudes(numElements);
    float minMag = 1e20f;
    float maxMag = -1e20f;

    // Task 4.2: Magnitude Calculation using OpenMP
    // Map with reduction to find min/max values in a single parallel sweep
    #pragma omp parallel
    {
        float localMin = 1e20f;
        float localMax = -1e20f;

        #pragma omp for nowait
        for (size_t i = 0; i < numElements; ++i) {
            // Absolute magnitude: sqrt(r^2 + i^2)
            // Divide by 'lines' to scale for the unnormalized inverse cuFFT (Azimuth IFFT)
            float realVal = h_data[i].r;
            float imagVal = h_data[i].i;
            float mag = std::sqrt(realVal * realVal + imagVal * imagVal) / lines;
            magnitudes[i] = mag;

            if (mag < localMin) localMin = mag;
            if (mag > localMax) localMax = mag;
        }

        #pragma omp critical
        {
            if (localMin < minMag) minMag = localMin;
            if (localMax > maxMag) maxMag = localMax;
        }
    }

    std::cout << "  - Computed magnitudes. Min: " << minMag << ", Max: " << maxMag << std::endl;

    // Prevent divide-by-zero if image is flat
    float range = maxMag - minMag;
    if (range < 1e-6f) {
        range = 1.0f;
    }

    // Allocate 1 byte per pixel for grayscale PGM
    std::vector<unsigned char> pixels(numElements);

    // Map intensities to 0-255 scale using OpenMP
    std::cout << "  - Normalizing intensities to 0-255 range..." << std::endl;
    #pragma omp parallel for
    for (size_t i = 0; i < numElements; ++i) {
        float normalized = (magnitudes[i] - minMag) / range;
        
        // Apply a small power scaling (gamma correction) to enhance radar contrast (e.g. gamma = 0.5)
        normalized = std::pow(normalized, 0.5f);
        
        pixels[i] = static_cast<unsigned char>(std::min(std::max(normalized * 255.0f, 0.0f), 255.0f));
    }

    // Task 4.3: Save as PGM (P5 format)
    std::cout << "  - Writing image to PGM file: " << outputPath << "..." << std::endl;
    std::ofstream file(outputPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output image file: " << outputPath << std::endl;
        return false;
    }

    // PGM P5 header: width (samples) and height (lines)
    file << "P5\n" << samples << " " << lines << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), numElements);
    file.close();

    double tEnd = omp_get_wtime();
    synthesisSecs = tEnd - tStart;
    
    std::cout << "Image synthesis completed in " << synthesisSecs << " seconds." << std::endl;
    return true;
}
