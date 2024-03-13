#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

SYCL_EXTERNAL void magmablas_zelementary_multiplication_kernel(
    magma_int_t n, magmaDoubleComplex *dA, magma_int_t offsetA,
    magma_int_t ldda, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *dv, magma_int_t offsetv, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *u1, magmaDoubleComplex *u2, magmaDoubleComplex *v1,
    magmaDoubleComplex *v2);

SYCL_EXTERNAL void
magmablas_zapply_vector_kernel(magma_int_t n, magmaDoubleComplex *du,
                               magma_int_t offsetu, magmaDoubleComplex *db,
                               magma_int_t offsetb, sycl::nd_item<3> item_ct1);

SYCL_EXTERNAL void magmablas_zapply_transpose_vector_kernel(
    magma_int_t n, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *db, magma_int_t offsetb, sycl::nd_item<3> item_ct1);

// =============================================================================
// batched prototypes

SYCL_EXTERNAL void magmablas_zelementary_multiplication_kernel_batched(
    magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t offsetA,
    magma_int_t ldda, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *dv, magma_int_t offsetv, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *u1, magmaDoubleComplex *u2, magmaDoubleComplex *v1,
    magmaDoubleComplex *v2);

SYCL_EXTERNAL void magmablas_zapply_vector_kernel_batched(
    magma_int_t n, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex **db_array, magma_int_t offsetb,
    sycl::nd_item<3> item_ct1);

SYCL_EXTERNAL void magmablas_zapply_transpose_vector_kernel_batched(
    magma_int_t n, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex **db_array, magma_int_t offsetb,
    sycl::nd_item<3> item_ct1);

#endif // ZGERBT_H
