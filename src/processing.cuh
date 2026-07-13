#ifndef PROCESSING_CUH
#define PROCESSING_CUH

#include "types.h"

// Run the full GPU processing pipeline (Phase 3)
bool runGPUProcessing(ComplexFloat* h_data, int lines, int samples, 
                      double centerFrequency, double slantRange, double prf,
                      double& totalGPUSecs);

#endif // PROCESSING_CUH
