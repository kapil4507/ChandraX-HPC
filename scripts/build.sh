#!/bin/bash
# Compilation script for DFSAR synthesis pipeline

# Clean up previous builds
rm -f app_dfsar

echo "Detecting compiler environment..."

# Check if nvcc is available
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) detected. Compiling for GPU-enabled target..."
    nvcc -O3 -Xcompiler -fopenmp -std=c++11 -allow-unsupported-compiler \
        src/main.cu \
        src/parser.cpp \
        src/ingestion.cpp \
        src/processing.cu \
        src/synthesis.cpp \
        -o app_dfsar \
        -lcufft
else
    echo "nvcc not found. Falling back to g++ (CPU-only compilation)..."
    if command -v g++ &> /dev/null; then
        # Compile main.cu as C++ code, omitting processing.cu
        g++ -O3 -fopenmp -std=c++11 -x c++ \
            src/main.cu \
            src/parser.cpp \
            src/ingestion.cpp \
            src/synthesis.cpp \
            -o app_dfsar
    else
        echo "Error: Neither nvcc nor g++ was found in your PATH."
        exit 1
    fi
fi

if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "Compilation successful! Executable: ./app_dfsar"
    echo "----------------------------------------"
else
    echo "Compilation failed."
    exit 1
fi
