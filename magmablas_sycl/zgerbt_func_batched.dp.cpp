/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Adrien Remy
       @author Azzam Haidar
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "zgerbt.h"

#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64

/***************************************************************************//**
    Purpose
    -------
    ZPRBT_MVT compute B = UTB to randomize B

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in]
    du     COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V

    @param[in,out]
    db     COMPLEX_16 array, dimension (n)
            The n vector db computed by ZGESV_NOPIV_GPU
            On exit db = du*db

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C" void
magmablas_zprbt_mtv_batched(
    magma_int_t n,
    magmaDoubleComplex *du, magmaDoubleComplex **db_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t threads = block_length;
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(int i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(1, ibatch, magma_ceildiv(n, 4 * block_length));

        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                  sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    magmablas_zapply_transpose_vector_kernel_batched(
                        n / 2, du, n, db_array + i, 0, item_ct1);
                });
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                  sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    magmablas_zapply_transpose_vector_kernel_batched(
                        n / 2, du, n + n / 2, db_array + i, n / 2, item_ct1);
                });

        threads = block_length;
        grid = sycl::range<3>(1, 1, magma_ceildiv(n, 2 * block_length));
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                  sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    magmablas_zapply_transpose_vector_kernel_batched(
                        n, du, 0, db_array + i, 0, item_ct1);
                });
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZPRBT_MV compute B = VB to obtain the non randomized solution

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in,out]
    db      COMPLEX_16 array, dimension (n)
            The n vector db computed by ZGESV_NOPIV_GPU
            On exit db = dv*db

    @param[in]
    dv      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C" void
magmablas_zprbt_mv_batched(
    magma_int_t n,
    magmaDoubleComplex *dv, magmaDoubleComplex **db_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t threads = block_length;
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(1, ibatch, magma_ceildiv(n, 2 * block_length));
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                  sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    magmablas_zapply_vector_kernel_batched(
                        n, dv, 0, db_array + i, 0, item_ct1);
                });

        threads = block_length;
        grid = sycl::range<3>(1, 1, magma_ceildiv(n, 4 * block_length));
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                  sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    magmablas_zapply_vector_kernel_batched(
                        n / 2, dv, n, db_array + i, 0, item_ct1);
                });
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                  sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    magmablas_zapply_vector_kernel_batched(
                        n / 2, dv, n + n / 2, db_array + i, n / 2, item_ct1);
                });
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZPRBT randomize a square general matrix using partial randomized transformation

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns and rows of the matrix dA.  n >= 0.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (n,ldda)
            The n-by-n matrix dA
            On exit dA = duT*dA*d_V

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDA >= max(1,n).

    @param[in]
    du      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix U

    @param[in]
    dv      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C" void
magmablas_zprbt_batched(
    magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex *du, magmaDoubleComplex *dv,
    magma_int_t batchCount, magma_queue_t queue)
{
    du += ldda;
    dv += ldda;

    sycl::range<3> threads(1, block_width, block_height);
    sycl::range<3> threads2(1, block_width, block_height);
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, magma_ceildiv(n, 4 * block_width),
                            magma_ceildiv(n, 4 * block_height));

        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u1_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u2_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v1_acc_ct1(sycl::range<1>(block_width), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v2_acc_ct1(sycl::range<1>(block_width), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmablas_zelementary_multiplication_kernel_batched(
                            n / 2, dA_array + i, 0, ldda, du, 0, dv, 0,
                            item_ct1, u1_acc_ct1.get_pointer(),
                            u2_acc_ct1.get_pointer(), v1_acc_ct1.get_pointer(),
                            v2_acc_ct1.get_pointer());
                    });
            });
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u1_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u2_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v1_acc_ct1(sycl::range<1>(block_width), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v2_acc_ct1(sycl::range<1>(block_width), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmablas_zelementary_multiplication_kernel_batched(
                            n / 2, dA_array + i, ldda * n / 2, ldda, du, 0, dv,
                            n / 2, item_ct1, u1_acc_ct1.get_pointer(),
                            u2_acc_ct1.get_pointer(), v1_acc_ct1.get_pointer(),
                            v2_acc_ct1.get_pointer());
                    });
            });
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u1_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u2_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v1_acc_ct1(sycl::range<1>(block_width), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v2_acc_ct1(sycl::range<1>(block_width), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmablas_zelementary_multiplication_kernel_batched(
                            n / 2, dA_array + i, n / 2, ldda, du, n / 2, dv, 0,
                            item_ct1, u1_acc_ct1.get_pointer(),
                            u2_acc_ct1.get_pointer(), v1_acc_ct1.get_pointer(),
                            v2_acc_ct1.get_pointer());
                    });
            });
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u1_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u2_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v1_acc_ct1(sycl::range<1>(block_width), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v2_acc_ct1(sycl::range<1>(block_width), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmablas_zelementary_multiplication_kernel_batched(
                            n / 2, dA_array + i, ldda * n / 2 + n / 2, ldda, du,
                            n / 2, dv, n / 2, item_ct1,
                            u1_acc_ct1.get_pointer(), u2_acc_ct1.get_pointer(),
                            v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
                    });
            });

        sycl::range<3> grid2(ibatch, magma_ceildiv(n, 2 * block_width),
                             magma_ceildiv(n, 2 * block_height));
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u1_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    u2_acc_ct1(sycl::range<1>(block_height), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v1_acc_ct1(sycl::range<1>(block_width), cgh);
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    v2_acc_ct1(sycl::range<1>(block_width), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid2 * threads2, threads2),
                    [=](sycl::nd_item<3> item_ct1) {
                        magmablas_zelementary_multiplication_kernel_batched(
                            n, dA_array + i, 0, ldda, du, -ldda, dv, -ldda,
                            item_ct1, u1_acc_ct1.get_pointer(),
                            u2_acc_ct1.get_pointer(), v1_acc_ct1.get_pointer(),
                            v2_acc_ct1.get_pointer());
                    });
            });
    }
}
