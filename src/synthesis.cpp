#include "synthesis.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <algorithm>

using namespace std;

bool synthesizeImage(const ComplexFloat* h_data, int lines, int samples, const string& outputPath, double& synthesisSecs) {
    double tStart = omp_get_wtime();
    int aspectFactor = 14; // Downsampling factor for azimuth
    int outLines = lines / aspectFactor;
    size_t outElements = static_cast<size_t>(outLines) * samples;

    vector<float> magnitudes(outElements);
    double sum = 0.0;

    // Task 4.2: Magnitude Calculation, Aspect Ratio Downsampling, dB Scaling
    #pragma omp parallel for reduction(+:sum)
    for (int row = 0; row < outLines; ++row) {
        for (int col = 0; col < samples; ++col) {
            float magSum = 0.0f;
            for (int k = 0; k < aspectFactor; ++k) {
                int origRow = row * aspectFactor + k;
                size_t origIdx = static_cast<size_t>(origRow) * samples + col;
                float realVal = h_data[origIdx].r;
                float imagVal = h_data[origIdx].i;
                magSum += sqrt(realVal * realVal + imagVal * imagVal) / lines;
            }
            float avgMag = magSum / aspectFactor;
            float val_dB = 20.0f * log10(avgMag + 1e-6f); // Decibel conversion
            
            size_t outIdx = static_cast<size_t>(row) * samples + col;
            magnitudes[outIdx] = val_dB;
            sum += val_dB;
        }
    }

    float mean = static_cast<float>(sum / outElements);

    // Compute standard deviation
    double sq_sum = 0.0;
    #pragma omp parallel for reduction(+:sq_sum)
    for (size_t i = 0; i < outElements; ++i) {
        float diff = magnitudes[i] - mean;
        sq_sum += diff * diff;
    }
    float stddev = sqrt(static_cast<float>(sq_sum / outElements));

    // Apply 2.5-sigma clipping (standard practice for high-dynamic-range radar imagery)
    // This removes bright noise spikes/calibration frame outliers and stretches active pixels
    float minClip = mean - 2.5f * stddev;
    float maxClip = mean + 2.5f * stddev;
    
    if (minClip < 0.0f) minClip = 0.0f;
    if (maxClip <= minClip) maxClip = minClip + 1e-5f; // Prevent divide by zero

    float range = maxClip - minClip;

    cout << "  - Image Stats: Mean = " << mean << ", StdDev = " << stddev << endl;
    cout << "  - Scaling Range (2.5-sigma): [" << minClip << ", " << maxClip << "]" << endl;

    // Allocate 1 byte per pixel for grayscale PGM
    vector<unsigned char> pixels(outElements);

    // Map intensities to 0-255 scale using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < outElements; ++i) {
        float val = magnitudes[i];
        if (val > maxClip) val = maxClip;
        if (val < minClip) val = minClip;
        
        float normalized = (val - minClip) / range;
        pixels[i] = static_cast<unsigned char>(normalized * 255.0f);
    }

    // Task 4.3: Save as PGM (P5 format)
    cout << "  - Writing image to PGM file: " << outputPath << "..." << endl;
    ofstream file(outputPath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open output image file: " << outputPath << endl;
        return false;
    }

    // PGM P5 header
    file << "P5\n" << samples << " " << outLines << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), outElements);
    file.close();

    double tEnd = omp_get_wtime();
    synthesisSecs = tEnd - tStart;
    
    cout << "Image synthesis completed in " << synthesisSecs << " seconds." << endl;
    return true;
}
