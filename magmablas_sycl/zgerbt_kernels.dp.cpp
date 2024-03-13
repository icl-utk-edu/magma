/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c


       @author Adrien REMY
*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "zgerbt.h"

#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64


/******************************************************************************/
static void 
magmablas_zelementary_multiplication_devfunc(
    magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda, 
    magmaDoubleComplex *du, 
    magmaDoubleComplex *dv, sycl::nd_item<3> item_ct1, magmaDoubleComplex *u1,
    magmaDoubleComplex *u2, magmaDoubleComplex *v1, magmaDoubleComplex *v2)
{    
    magma_int_t idx, idy;

    idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
    idy = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);

    if ((idx < n/2) && (idy < n/2)) {
        dA += idx + idy * ldda;

        magmaDoubleComplex a00, a10, a01, a11, b1, b2, b3, b4;

        du += idx;
        dv += idy;

        u1[item_ct1.get_local_id(2)] = du[0];
        u2[item_ct1.get_local_id(2)] = du[n / 2];
        v1[item_ct1.get_local_id(1)] = dv[0];
        v2[item_ct1.get_local_id(1)] = dv[n / 2];

        /*
        DPCT1065:507: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        a00 = dA[0];
        a01 = dA[ldda*n/2];
        a10 = dA[n/2];
        a11 = dA[ldda*n/2+n/2];

        b1 = a00 + a01;
        b2 = a10 + a11;
        b3 = a00 - a01;
        b4 = a10 - a11;

        dA[0] = u1[item_ct1.get_local_id(2)] * v1[item_ct1.get_local_id(1)] *
                (b1 + b2);
        dA[ldda * n / 2] = u1[item_ct1.get_local_id(2)] *
                           v2[item_ct1.get_local_id(1)] * (b3 + b4);
        dA[n / 2] = u2[item_ct1.get_local_id(2)] *
                    v1[item_ct1.get_local_id(1)] * (b1 - b2);
        dA[ldda * n / 2 + n / 2] = u2[item_ct1.get_local_id(2)] *
                                   v2[item_ct1.get_local_id(1)] * (b3 - b4);
    }
}


/******************************************************************************/
SYCL_EXTERNAL void magmablas_zelementary_multiplication_kernel(
    magma_int_t n, magmaDoubleComplex *dA, magma_int_t offsetA,
    magma_int_t ldda, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *dv, magma_int_t offsetv, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *u1, magmaDoubleComplex *u2, magmaDoubleComplex *v1,
    magmaDoubleComplex *v2)
{
    magmablas_zelementary_multiplication_devfunc(n, dA + offsetA, ldda,
                                                 du + offsetu, dv + offsetv,
                                                 item_ct1, u1, u2, v1, v2);
}


/******************************************************************************/
SYCL_EXTERNAL void magmablas_zelementary_multiplication_kernel_batched(
    magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t offsetA,
    magma_int_t ldda, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *dv, magma_int_t offsetv, sycl::nd_item<3> item_ct1,
    magmaDoubleComplex *u1, magmaDoubleComplex *u2, magmaDoubleComplex *v1,
    magmaDoubleComplex *v2)
{
    int batchid = item_ct1.get_group(0);
    magmablas_zelementary_multiplication_devfunc(
        n, dA_array[batchid] + offsetA, ldda, du + offsetu, dv + offsetv,
        item_ct1, u1, u2, v1, v2);
}


/******************************************************************************/
static void 
magmablas_zapply_vector_devfunc(
    magma_int_t n,
    magmaDoubleComplex *du, magmaDoubleComplex *db, sycl::nd_item<3> item_ct1)
{
    magma_int_t idx;

    idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);

    if (idx < n/2) {
        du += idx;
        db += idx;

        magmaDoubleComplex a1,a2;

        a1 = du[0]*db[0];
        a2 = du[n/2]*db[n/2];

        db[0] = a1 + a2;
        db[n/2] = a1 -a2;
    }
}


/******************************************************************************/
SYCL_EXTERNAL void
magmablas_zapply_vector_kernel(magma_int_t n, magmaDoubleComplex *du,
                               magma_int_t offsetu, magmaDoubleComplex *db,
                               magma_int_t offsetb, sycl::nd_item<3> item_ct1)
{
    magmablas_zapply_vector_devfunc(n, du + offsetu, db + offsetb, item_ct1);
}


/******************************************************************************/
SYCL_EXTERNAL void magmablas_zapply_vector_kernel_batched(
    magma_int_t n, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex **db_array, magma_int_t offsetb,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(1);
    magmablas_zapply_vector_devfunc(n, du + offsetu,
                                    db_array[batchid] + offsetb, item_ct1);
}


/******************************************************************************/
static void 
magmablas_zapply_transpose_vector_devfunc(
    magma_int_t n,
    magmaDoubleComplex *du,magmaDoubleComplex *db , sycl::nd_item<3> item_ct1)
{
    magma_int_t idx;

    idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);

    if (idx < n/2) {
        du += idx;
        db += idx;

        magmaDoubleComplex a1,a2;

        a1 = db[0] + db[n/2];
        a2 = db[0] - db[n/2];

        db[0] = du[0]*a1;
        db[n/2] = du[n/2]*a2;
    }
}


/******************************************************************************/
SYCL_EXTERNAL void magmablas_zapply_transpose_vector_kernel(
    magma_int_t n, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex *db, magma_int_t offsetb, sycl::nd_item<3> item_ct1)
{
    magmablas_zapply_transpose_vector_devfunc(n, du + offsetu, db + offsetb,
                                              item_ct1);
}


/******************************************************************************/
SYCL_EXTERNAL void magmablas_zapply_transpose_vector_kernel_batched(
    magma_int_t n, magmaDoubleComplex *du, magma_int_t offsetu,
    magmaDoubleComplex **db_array, magma_int_t offsetb,
    sycl::nd_item<3> item_ct1)
{
    int batchid = item_ct1.get_group(1);
    magmablas_zapply_transpose_vector_devfunc(
        n, du + offsetu, db_array[batchid] + offsetb, item_ct1);
}
