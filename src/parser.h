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
};

// Parses a PDS4 XML label file and extracts dimensions
MatrixDimensions parseXML(const std::string& xmlPath);

#endif // PARSER_H
