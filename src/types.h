#ifndef TYPES_H
#define TYPES_H

// Custom complex float structure.
// Binary compatible with cuComplex / cufftComplex (two consecutive floats representing real and imaginary parts).
struct ComplexFloat {
    float r; // Real part
    float i; // Imaginary part
};

#endif // TYPES_H
