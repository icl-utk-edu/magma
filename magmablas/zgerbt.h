/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Adrien Remy
       @author Azzam Haidar

       Definitions used in zgerbt.cu zgerbt_batched.cu
*/

#ifndef ZGERBT_H
#define ZGERBT_H

// =============================================================================
// classical prototypes

__global__ void
magmablas_zelementary_multiplication_kernel(
    magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t offsetA, magma_int_t ldda,
    magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *dv, magma_int_t offsetv);

__global__ void
magmablas_zelementary_multiplication_v2_kernel_batched(
    int Am, int An,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex *du, magma_int_t Ui,
    magmaDoubleComplex *dv, magma_int_t Vi);

__global__ void
magmablas_zapply_vector_kernel(
    magma_int_t n,
    magmaDoubleComplex *du, magma_int_t offsetu,  magmaDoubleComplex *db, magma_int_t offsetb );

__global__ void
magmablas_zapply_transpose_vector_kernel(
    magma_int_t n,
    magmaDoubleComplex *du, magma_int_t offsetu, magmaDoubleComplex *db, magma_int_t offsetb );

// =============================================================================
// batched prototypes

__global__ void
magmablas_zelementary_multiplication_kernel_batched(
    magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t offsetA, magma_int_t ldda,
    magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *dv, magma_int_t offsetv);

__global__ void
magmablas_zapply_vector_kernel_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *du, magma_int_t offsetu, magmaDoubleComplex **db_array, magma_int_t lddb, magma_int_t offsetb );

__global__ void
magmablas_zapply_transpose_vector_kernel_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *du, magma_int_t offsetu, magmaDoubleComplex **db_array, magma_int_t lddb, magma_int_t offsetb );

#endif // ZGERBT_H
