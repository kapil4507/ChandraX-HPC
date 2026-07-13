#include "processing.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>
#include <cmath>

using namespace std;

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << endl; \
            return false; \
        } \
    } while (0)

// Helper macro for checking cuFFT errors
#define CHECK_CUFFT(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            cerr << "cuFFT Error at " << __FILE__ << ":" << __LINE__ \
                      << " - Code: " << result << endl; \
            return false; \
        } \
    } while (0)

// Custom CUDA Kernel for Range focusing (Matched Filter)
// Applies quadratic phase correction in the range frequency domain.
__global__ void rangeMatchedFilterKernel(cufftComplex* d_data, int lines, int samples, 
                                         float Kr, float samplingRate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < lines && col < samples) {
        int idx = row * samples + col;
        
        // Compute range frequency corresponding to this sample
        float f = (col < samples / 2) ? 
                  col * (samplingRate / samples) : 
                  (col - samples) * (samplingRate / samples);
                  
        // Range phase correction: phi = pi * f^2 / Kr
        float phase = 3.14159265f * f * f / Kr;
        
        // Matched filter conjugate reference: exp(j * phase)
        float ref_x = cosf(phase);
        float ref_y = sinf(phase);
        
        // Complex multiplication: data * conjugate(reference)
        // Wait, the range matched filter requires the complex conjugate, so exp(j * phase) is correct if the chirp was exp(-j * phase)
        cufftComplex val = d_data[idx];
        cufftComplex res;
        res.x = val.x * ref_x - val.y * ref_y;
        res.y = val.x * ref_y + val.y * ref_x;
        
        // Normalize range IFFT since cuFFT unnormalized
        res.x /= samples;
        res.y /= samples;
        
        d_data[idx] = res;
    }
}

// Custom CUDA Kernel for Azimuth Doppler focusing (Matched Filter)
// Applies quadratic phase correction in the Doppler frequency domain.
__global__ void azimuthMatchedFilterKernel(cufftComplex* d_data, int lines, int samples, 
                                           float Ka, float prf) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < lines && col < samples) {
        int idx = row * samples + col;
        
        // Compute azimuth (Doppler) frequency corresponding to this line
        float f = (row < lines / 2) ? 
                  row * (prf / lines) : 
                  (row - lines) * (prf / lines);
                  
        // Doppler phase correction: phi = pi * f^2 / Ka
        float phase = 3.14159265f * f * f / Ka;
        
        // Matched filter conjugate reference: exp(j * phase) = cos(phase) + j * sin(phase)
        float ref_x = cosf(phase);
        float ref_y = sinf(phase);
        
        // Complex multiplication: (val.x + i * val.y) * (ref_x + i * ref_y)
        cufftComplex val = d_data[idx];
        cufftComplex res;
        res.x = val.x * ref_x - val.y * ref_y;
        res.y = val.x * ref_y + val.y * ref_x;
        
        d_data[idx] = res;
    }
}

bool runGPUProcessing(ComplexFloat* h_data, int lines, int samples, 
                      double centerFrequency, double slantRange, double prf,
                      double rangeBandwidth, double pulseDuration, double& totalGPUSecs) {
    double tStart = omp_get_wtime();

    size_t numElements = static_cast<size_t>(lines) * samples;
    size_t dataSizeBytes = numElements * sizeof(cufftComplex);

    cufftComplex* d_data = nullptr;

    cout << "[Phase 3] Initializing GPU Processing..." << endl;
    
    // Task 3.1: Host to Device Transfer
    cout << "  - Allocating GPU device memory (cudaMalloc)..." << endl;
    CHECK_CUDA(cudaMalloc(&d_data, dataSizeBytes));

    cout << "  - Copying radar matrix to GPU (cudaMemcpy)..." << endl;
    CHECK_CUDA(cudaMemcpy(d_data, h_data, dataSizeBytes, cudaMemcpyHostToDevice));

    // Task 3.2a: Range Compression (Row-wise FFT)
    cout << "  - Executing Forward 1D cuFFT (Range) across rows..." << endl;
    int rankR = 1;
    int nR[1] = { samples };
    int inembedR[1] = { samples };
    int istrideR = 1;
    int idistR = samples;
    int onembedR[1] = { samples };
    int ostrideR = 1;
    int odistR = samples;
    
    cufftHandle rangePlan;
    CHECK_CUFFT(cufftPlanMany(&rangePlan, rankR, nR, 
                              inembedR, istrideR, idistR, 
                              onembedR, ostrideR, odistR, 
                              CUFFT_C2C, lines));
                              
    CHECK_CUFFT(cufftExecC2C(rangePlan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute Range Matched Filter
    float Kr = static_cast<float>(rangeBandwidth / pulseDuration);
    float fs = 83.33e6f; // Standard DFSAR ADC sampling rate 83.33 MHz
    cout << "  - Executing Range Matched Filter Kernel (Kr=" << Kr << " Hz/s)..." << endl;
    
    dim3 blockDimR(16, 16);
    dim3 gridDimR((samples + blockDimR.x - 1) / blockDimR.x, 
                  (lines + blockDimR.y - 1) / blockDimR.y);
    rangeMatchedFilterKernel<<<gridDimR, blockDimR>>>(d_data, lines, samples, Kr, fs);
    CHECK_CUDA(cudaDeviceSynchronize());

    cout << "  - Executing Inverse 1D cuFFT (Range) across rows..." << endl;
    CHECK_CUFFT(cufftExecC2C(rangePlan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUFFT(cufftDestroy(rangePlan));

    // Task 3.2b: 1D Forward FFT along columns (Azimuth FFT)
    cout << "  - Executing Forward 1D cuFFT (Azimuth) across columns..." << endl;
    
    // Column-wise 1D FFT using cufftPlanMany:
    // We execute a batch of 'samples' transforms, each of size 'lines'.
    // Elements of a single transform are separated by 'samples' items (stride = samples).
    int rank = 1;
    int n[1] = { lines };
    int inembed[1] = { lines };
    int istride = samples;
    int idist = 1;
    int onembed[1] = { lines };
    int ostride = samples;
    int odist = 1;
    
    cufftHandle azimuthPlan;
    CHECK_CUFFT(cufftPlanMany(&azimuthPlan, rank, n, 
                              inembed, istride, idist, 
                              onembed, ostride, odist, 
                              CUFFT_C2C, samples));
    
    CHECK_CUFFT(cufftExecC2C(azimuthPlan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Task 3.3: Doppler Focusing Custom CUDA Kernel (Matched Filter)
    // Spacecraft velocity V for low lunar orbit (~84km altitude) is ~1640 m/s
    float V = 1640.0f;
    float lambda = 299792458.0f / static_cast<float>(centerFrequency);
    float Ka = (2.0f * V * V) / (lambda * static_cast<float>(slantRange));
    
    cout << "  - Calculated Doppler parameters:" << endl;
    cout << "    - Wavelength (lambda): " << lambda << " m" << endl;
    cout << "    - Azimuth FM rate (Ka): " << Ka << " Hz/s" << endl;
    cout << "  - Executing Azimuth Matched Filter Doppler focusing kernel..." << endl;
    
    dim3 blockDim(16, 16);
    dim3 gridDim((samples + blockDim.x - 1) / blockDim.x, 
                 (lines + blockDim.y - 1) / blockDim.y);
    azimuthMatchedFilterKernel<<<gridDim, blockDim>>>(d_data, lines, samples, Ka, static_cast<float>(prf));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Task 3.4: 1D Inverse FFT along columns (Azimuth IFFT)
    cout << "  - Executing Inverse 1D cuFFT (Azimuth) across columns..." << endl;
    CHECK_CUFFT(cufftExecC2C(azimuthPlan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clean up plan
    CHECK_CUFFT(cufftDestroy(azimuthPlan));

    // Task 4.1: Device to Host Transfer
    cout << "  - Transferring focused matrix back to Host (cudaMemcpy)..." << endl;
    CHECK_CUDA(cudaMemcpy(h_data, d_data, dataSizeBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    cout << "  - Freeing GPU device memory..." << endl;
    CHECK_CUDA(cudaFree(d_data));

    double tEnd = omp_get_wtime();
    totalGPUSecs = tEnd - tStart;
    
    cout << "GPU Processing finished successfully in " << totalGPUSecs << " seconds." << endl;
    return true;
}
