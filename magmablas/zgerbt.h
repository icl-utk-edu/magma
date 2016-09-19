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
    magma_int_t n,
    magmaDoubleComplex *du, magma_int_t offsetu, magmaDoubleComplex **db_array, magma_int_t offsetb );

__global__ void 
magmablas_zapply_transpose_vector_kernel_batched(
    magma_int_t n,
    magmaDoubleComplex *du, magma_int_t offsetu, magmaDoubleComplex **db_array, magma_int_t offsetb );

#endif // ZGERBT_H
