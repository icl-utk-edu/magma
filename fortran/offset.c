#include "magma.h"

#include <stdio.h>

// -----------------------------------------------------------------------------
// here, i is Fortran 1-based index
float* magma_soffset_1d(
    float* x, magma_int_t inc,
    magma_int_t i )
{
    return x + (i-1)*inc;
}

double* magma_doffset_1d(
    double* x, magma_int_t inc,
    magma_int_t i )
{
    return x + (i-1)*inc;
}

magmaFloatComplex* magma_coffset_1d(
    magmaFloatComplex* x, magma_int_t inc,
    magma_int_t i )
{
    return x + (i-1)*inc;
}

magmaDoubleComplex* magma_zoffset_1d(
    magmaDoubleComplex* x, magma_int_t inc,
    magma_int_t i )
{
    return x + (i-1)*inc;
}

magma_int_t* magma_ioffset_1d(
    magma_int_t* x, magma_int_t inc,
    magma_int_t i )
{
    return x + (i-1)*inc;
}


// -----------------------------------------------------------------------------
// here, i and j are Fortran 1-based indices
float* magma_soffset_2d(
    float* A, magma_int_t lda,
    magma_int_t i, magma_int_t j )
{
    return A + (i-1) + (j-1)*lda;
}

double* magma_doffset_2d(
    double* A, magma_int_t lda,
    magma_int_t i, magma_int_t j )
{
    return A + (i-1) + (j-1)*lda;
}

magmaFloatComplex* magma_coffset_2d(
    magmaFloatComplex* A, magma_int_t lda,
    magma_int_t i, magma_int_t j )
{
    return A + (i-1) + (j-1)*lda;
}

magmaDoubleComplex* magma_zoffset_2d(
    magmaDoubleComplex* A, magma_int_t lda,
    magma_int_t i, magma_int_t j )
{
    return A + (i-1) + (j-1)*lda;
}

magma_int_t* magma_ioffset_2d(
    magma_int_t* A, magma_int_t lda,
    magma_int_t i, magma_int_t j )
{
    return A + (i-1) + (j-1)*lda;
}
