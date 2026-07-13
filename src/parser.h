#ifndef PARSER_H
#define PARSER_H

#include <string>

struct MatrixDimensions {
    int lines = 0;              // Total number of lines in raw file (e.g. 509909)
    int totalLineElements = 0;  // Total bytes per line (e.g. 4885)
    int samplesPerEcho = 0;     // Number of echo samples per line (e.g. 1024)
    int numPols = 0;            // Number of polarizations (e.g. 2)
    std::string dataType;       // e.g., "SignedByte"
    std::string fileName;       // e.g., "ch2_sar_nrxl_..."
    double centerFrequency = 1.25e9;  // radar_center_frequency (Hz)
    double slantRange = 94287.05;     // slant_range_near_edge (m)
    double prf = 2596.38;             // pulse_repetition_frequency (Hz)
    int validSamples = 768;           // Active radar samples per echo (cropped from 1024)
    double rangeBandwidth = 75.0e6;   // Range bandwidth (Hz)
    double pulseDuration = 65.0e-6;   // Range pulse duration (s)
};

// Parses a PDS4 XML label file and extracts dimensions
MatrixDimensions parseXML(const std::string& xmlPath);

#endif // PARSER_H
