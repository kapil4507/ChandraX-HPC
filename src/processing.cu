#include "processing.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>
#include <cmath>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while (0)

// Helper macro for checking cuFFT errors
#define CHECK_CUFFT(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT Error at " << __FILE__ << ":" << __LINE__ \
                      << " - Code: " << result << std::endl; \
            return false; \
        } \
    } while (0)

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
                      double& totalGPUSecs) {
    double tStart = omp_get_wtime();

    size_t numElements = static_cast<size_t>(lines) * samples;
    size_t dataSizeBytes = numElements * sizeof(cufftComplex);

    cufftComplex* d_data = nullptr;

    std::cout << "[Phase 3] Initializing GPU Processing..." << std::endl;
    
    // Task 3.1: Host to Device Transfer
    std::cout << "  - Allocating GPU device memory (cudaMalloc)..." << std::endl;
    CHECK_CUDA(cudaMalloc(&d_data, dataSizeBytes));

    std::cout << "  - Copying radar matrix to GPU (cudaMemcpy)..." << std::endl;
    CHECK_CUDA(cudaMemcpy(d_data, h_data, dataSizeBytes, cudaMemcpyHostToDevice));

    // Task 3.2: 1D Forward FFT along columns (Azimuth FFT)
    std::cout << "  - Executing Forward 1D cuFFT (Azimuth) across columns..." << std::endl;
    
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
    
    std::cout << "  - Calculated Doppler parameters:" << std::endl;
    std::cout << "    - Wavelength (lambda): " << lambda << " m" << std::endl;
    std::cout << "    - Azimuth FM rate (Ka): " << Ka << " Hz/s" << std::endl;
    std::cout << "  - Executing Azimuth Matched Filter Doppler focusing kernel..." << std::endl;
    
    dim3 blockDim(16, 16);
    dim3 gridDim((samples + blockDim.x - 1) / blockDim.x, 
                 (lines + blockDim.y - 1) / blockDim.y);
    azimuthMatchedFilterKernel<<<gridDim, blockDim>>>(d_data, lines, samples, Ka, static_cast<float>(prf));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Task 3.4: 1D Inverse FFT along columns (Azimuth IFFT)
    std::cout << "  - Executing Inverse 1D cuFFT (Azimuth) across columns..." << std::endl;
    CHECK_CUFFT(cufftExecC2C(azimuthPlan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clean up plan
    CHECK_CUFFT(cufftDestroy(azimuthPlan));

    // Task 4.1: Device to Host Transfer
    std::cout << "  - Transferring focused matrix back to Host (cudaMemcpy)..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_data, d_data, dataSizeBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    std::cout << "  - Freeing GPU device memory..." << std::endl;
    CHECK_CUDA(cudaFree(d_data));

    double tEnd = omp_get_wtime();
    totalGPUSecs = tEnd - tStart;
    
    std::cout << "GPU Processing finished successfully in " << totalGPUSecs << " seconds." << std::endl;
    return true;
}
