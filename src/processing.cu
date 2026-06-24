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

// Custom CUDA Kernel for Matched Filter element-wise complex multiplication
// z1 * z2 = (ac - bd) + i(ad + bc)
__global__ void matchedFilterKernel(cufftComplex* d_data, const cufftComplex* d_ref, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        cufftComplex val = d_data[idx];
        cufftComplex ref = d_ref[idx];
        
        cufftComplex res;
        res.x = val.x * ref.x - val.y * ref.y;
        res.y = val.x * ref.y + val.y * ref.x;
        
        d_data[idx] = res;
    }
}

// Custom CUDA Kernel to generate synthetic reference chirp conjugate (matched filter)
__global__ void initSyntheticReference(cufftComplex* d_ref, int lines, int samples) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < lines && col < samples) {
        int idx = row * samples + col;
        
        // Compute synthetic quadratic phase chirp
        float t = static_cast<float>(col - samples / 2) / (samples / 2);
        float K = 25.0f; // Synthetic FM chirp rate
        float phase = 3.14159265f * K * t * t;
        
        // Conjugate of reference chirp is exp(-j * phase) = cos(phase) - j * sin(phase)
        d_ref[idx].x = cos(phase);
        d_ref[idx].y = -sin(phase);
    }
}

bool runGPUProcessing(ComplexFloat* h_data, int lines, int samples, double& totalGPUSecs) {
    double tStart = omp_get_wtime();

    size_t numElements = static_cast<size_t>(lines) * samples;
    size_t dataSizeBytes = numElements * sizeof(cufftComplex);

    cufftComplex* d_data = nullptr;
    cufftComplex* d_ref = nullptr;

    std::cout << "[Phase 3] Initializing GPU Processing..." << std::endl;
    
    // Task 3.1: Host to Device Transfer
    std::cout << "  - Allocating GPU device memory (cudaMalloc)..." << std::endl;
    CHECK_CUDA(cudaMalloc(&d_data, dataSizeBytes));
    CHECK_CUDA(cudaMalloc(&d_ref, dataSizeBytes));

    std::cout << "  - Copying radar matrix to GPU (cudaMemcpy)..." << std::endl;
    CHECK_CUDA(cudaMemcpy(d_data, h_data, dataSizeBytes, cudaMemcpyHostToDevice));

    // Initialize the synthetic matched filter reference on GPU
    std::cout << "  - Synthesizing Matched Filter Reference on GPU..." << std::endl;
    dim3 blockDimInit(16, 16);
    dim3 gridDimInit((samples + blockDimInit.x - 1) / blockDimInit.x, 
                     (lines + blockDimInit.y - 1) / blockDimInit.y);
    initSyntheticReference<<<gridDimInit, blockDimInit>>>(d_ref, lines, samples);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Task 3.2: First Pass FFT (Range)
    std::cout << "  - Executing Forward 1D cuFFT (Range) across rows..." << std::endl;
    cufftHandle rangePlan;
    // Plan a batch of 'lines' 1D transforms, each of size 'samples'
    CHECK_CUFFT(cufftPlan1d(&rangePlan, samples, CUFFT_C2C, lines));
    CHECK_CUFFT(cufftExecC2C(rangePlan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUFFT(cufftDestroy(rangePlan));

    // Task 3.3: Custom CUDA Kernel (Matched Filter)
    std::cout << "  - Executing Matched Filter custom CUDA kernel..." << std::endl;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    matchedFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_ref, numElements);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Task 3.4: Second Pass FFT (Azimuth)
    std::cout << "  - Executing Inverse 1D cuFFT (Azimuth) across columns..." << std::endl;
    
    // Column-wise 1D FFT using cufftPlanMany:
    // We execute a batch of 'samples' transforms, each of size 'lines'.
    // Elements of a single transform are separated by 'samples' items.
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
    
    CHECK_CUFFT(cufftExecC2C(azimuthPlan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUFFT(cufftDestroy(azimuthPlan));

    // Task 4.1: Device to Host Transfer (Phase 4.1)
    std::cout << "  - Transferring focused matrix back to Host (cudaMemcpy)..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_data, d_data, dataSizeBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    std::cout << "  - Freeing GPU device memory..." << std::endl;
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_ref));

    double tEnd = omp_get_wtime();
    totalGPUSecs = tEnd - tStart;
    
    std::cout << "GPU Processing finished successfully in " << totalGPUSecs << " seconds." << std::endl;
    return true;
}
