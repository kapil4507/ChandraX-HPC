#ifndef PROCESSING_CUH
#define PROCESSING_CUH

#include "types.h"

// Run the full GPU processing pipeline (Phase 3)
// Allocates GPU memory, copies data, runs Range FFT, custom Matched Filter kernel, Azimuth IFFT, and copies data back.
bool runGPUProcessing(ComplexFloat* h_data, int lines, int samples, double& totalGPUSecs);

#endif // PROCESSING_CUH
