#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include <string>
#include "types.h"

// Computes the magnitude of the complex elements, maps them to a 0-255 grayscale range,
// and saves the output as a PGM image file.
bool synthesizeImage(const ComplexFloat* h_data, int lines, int samples, const std::string& outputPath, double& synthesisSecs);

#endif // SYNTHESIS_H
