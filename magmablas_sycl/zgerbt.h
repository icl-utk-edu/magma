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
    int Am, int An,
    magmaDoubleComplex *dA, int Ai, int Aj, int ldda,
    magmaDoubleComplex *du, int Ui,
    magmaDoubleComplex *dv, int Vi, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *u1, magmaDoubleComplex *u2, magmaDoubleComplex *v1,
    magmaDoubleComplex *v2);

SYCL_EXTERNAL void
magmablas_zapply_vector_kernel(int n, int nrhs,
    magmaDoubleComplex *du, int offsetu,
    magmaDoubleComplex *db, int lddb,
    int offsetb, sycl::nd_item<3> item_ct1);

SYCL_EXTERNAL void magmablas_zapply_transpose_vector_kernel(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu,
    magmaDoubleComplex *db, int lddb, int offsetb, sycl::nd_item<3> item_ct1);

// =============================================================================
// batched prototypes

SYCL_EXTERNAL void magmablas_zelementary_multiplication_kernel_batched(
    int Am, int An,
    magmaDoubleComplex **dA_array, int Ai, int Aj, int ldda,
    magmaDoubleComplex *du, int Ui,
    magmaDoubleComplex *dv, int Vi, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *u1, magmaDoubleComplex *u2, magmaDoubleComplex *v1,
    magmaDoubleComplex *v2);

SYCL_EXTERNAL void magmablas_zapply_vector_kernel_batched(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu,
    magmaDoubleComplex **db_array, int lddb, int offsetb,
    sycl::nd_item<3> item_ct1);

SYCL_EXTERNAL void magmablas_zapply_transpose_vector_kernel_batched(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu,
    magmaDoubleComplex **db_array, int lddb, int offsetb,
    sycl::nd_item<3> item_ct1);

#endif // ZGERBT_H
