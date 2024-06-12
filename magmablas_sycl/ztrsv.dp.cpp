/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"


#define PRECISION_z

#define NB 256  //NB is the 1st level blocking in recursive blocking, NUM_THREADS is the 2ed level, NB=256, NUM_THREADS=64 is optimal for batched

#define NUM_THREADS 128 //64 //128

#define BLOCK_SIZE_N 128
#define DIM_X_N 128
#define DIM_Y_N 1

#define BLOCK_SIZE_T 32
#define DIM_X_T 16
#define DIM_Y_T 8

#include "ztrsv_template_device.dp.hpp"

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column


/******************************************************************************/
template< const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,
          const int TILE_SIZE, const int flag, const magma_uplo_t uplo,
          const magma_trans_t trans, const magma_diag_t diag >
void
ztrsv_notrans_kernel_outplace(
    int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    magmaDoubleComplex *b, int incb,
    magmaDoubleComplex *x, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    ztrsv_notrans_device<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans,
                         diag>(n, A, lda, b, incb, x, item_ct1, dpct_local);
}


/******************************************************************************/
template< const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,
          const int TILE_SIZE, const int flag, const magma_uplo_t uplo,
          const magma_trans_t trans, const magma_diag_t diag >
void
ztrsv_trans_kernel_outplace(
    int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    magmaDoubleComplex *b, int incb,
    magmaDoubleComplex *x, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    ztrsv_trans_device<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans,
                       diag>(n, A, lda, b, incb, x, item_ct1, dpct_local);
}


/******************************************************************************/
extern "C" void
magmablas_ztrsv_outofplace(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaDoubleComplex_ptr       b, magma_int_t incb,
    magmaDoubleComplex_ptr       x,
    magma_queue_t queue,
    magma_int_t flag=0)
{
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (n == 0)
        return;

    sycl::range<3> threads(1, 1, NUM_THREADS);
    sycl::range<3> blocks(1, 1, 1);
    /*
    DPCT1083:1513: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    size_t shmem = n * sizeof(magmaDoubleComplex);

    if (trans == MagmaNoTrans)
    {
        shmem += DIM_X_N * DIM_Y_N; // For the gemv that gets called in ztrsv_*_device
        if (uplo == MagmaUpper)
        {
            if (diag == MagmaNonUnit)
            {
                if (flag == 0) {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 0, MagmaUpper,
                                        MagmaNoTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 1, MagmaUpper,
                                        MagmaNoTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
            else if (diag == MagmaUnit)
            {
                if (flag == 0) {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 0, MagmaUpper,
                                        MagmaNoTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 1, MagmaUpper,
                                        MagmaNoTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
        }
        else //Lower
        {
            if (diag == MagmaNonUnit)
            {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 0, MagmaLower,
                                        MagmaNoTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 1, MagmaLower,
                                        MagmaNoTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
            else if (diag == MagmaUnit)
            {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 0, MagmaLower,
                                        MagmaNoTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_notrans_kernel_outplace<
                                        BLOCK_SIZE_N, DIM_X_N, DIM_Y_N,
                                        MagmaBigTileSize, 1, MagmaLower,
                                        MagmaNoTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
        }
    }
    else if (trans == MagmaTrans)
    {
        shmem += DIM_X_T * DIM_Y_T;
        if (uplo == MagmaUpper)
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaUpper,
                                        MagmaTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaUpper,
                                        MagmaTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaUpper,
                                        MagmaTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaUpper,
                                        MagmaTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
        }
        else
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaLower,
                                        MagmaTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaLower,
                                        MagmaTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaLower,
                                        MagmaTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaLower,
                                        MagmaTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
        }
    }
    else if (trans == MagmaConjTrans)
    {
        if (uplo == MagmaUpper)
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaUpper,
                                        MagmaConjTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaUpper,
                                        MagmaConjTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaUpper,
                                        MagmaConjTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaUpper,
                                        MagmaConjTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
        }
        else
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaLower,
                                        MagmaConjTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaLower,
                                        MagmaConjTrans, MagmaNonUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 0, MagmaLower,
                                        MagmaConjTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
                else {
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<uint8_t, 1>
                                dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(blocks * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    ztrsv_trans_kernel_outplace<
                                        BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,
                                        MagmaBigTileSize, 1, MagmaLower,
                                        MagmaConjTrans, MagmaUnit>(
                                        n, A, lda, b, incb, x, item_ct1,
                                        dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                });
                        });
                }
            }
        }
    }
}


/******************************************************************************/
/*
    README: flag decides if the ztrsv_outplace see an updated x or not. 0: No; other: Yes
    In recursive, flag must be nonzero except the 1st call
*/
extern "C" void
magmablas_ztrsv_recursive_outofplace(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaDoubleComplex_ptr       b, magma_int_t incb,
    magmaDoubleComplex_ptr       x,
    magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (n == 0)
        return;

    //Init x with zero
    //magmablas_zlaset( MagmaFull, n, incb, MAGMA_Z_ZERO, MAGMA_Z_ZERO, x, n, queue );

    magma_int_t col = n;

    if (trans == MagmaNoTrans)
    {
        for (magma_int_t i=0; i < n; i+= NB)
        {
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaUpper)
            {
                col -= jb;
                //assume x_array contains zero elements, magmablas_zgemv will cause slow down
                /*
                DPCT1064:1537: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                magma_zgemv(MagmaNoTrans, jb, i, MAGMA_Z_ONE, A(col, col + jb),
                            lda,
                            /*
                            DPCT1064:1538: Migrated make_cuDoubleComplex call is
                            used in a macro definition and is not valid for all
                            macro uses. Adjust the code.
                            */
                            x + col + jb, 1, MAGMA_Z_ONE, x + col, 1, queue);
            }
            else
            {
                col = i;
                /*
                DPCT1064:1539: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                magma_zgemv(MagmaNoTrans, jb, i, MAGMA_Z_ONE, A(col, 0), lda, x,
                            1, MAGMA_Z_ONE, x + col, 1, queue);
            }

            magmablas_ztrsv_outofplace( uplo, trans, diag, jb, A(col, col), lda, b+col, incb, x+col, queue, i );
        }
    }
    else
    {
        for (magma_int_t i=0; i < n; i += NB)
        {
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaLower)
            {
                col -= jb;

                magma_zgemv( MagmaConjTrans, i, jb, MAGMA_Z_ONE, A(col+jb, col), lda, x+col+jb, 1, MAGMA_Z_ONE, x+col, 1, queue );
            }
            else
            {
                col = i;
                
                magma_zgemv( MagmaConjTrans, i, jb, MAGMA_Z_ONE, A(0, col), lda, x, 1, MAGMA_Z_ONE, x+col, 1, queue );
            }
     
            magmablas_ztrsv_outofplace( uplo, trans, diag, jb, A(col, col), lda, b+col, incb, x+col, queue, i );
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ztrsv solves one of the matrix equations on gpu

        op(A)*x = B,   or
        x*op(A) = B,

    where alpha is a scalar, X and B are vectors, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The vector x is overwritten on b.

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    trans  magma_trans_t.
            On entry, trans specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n N specifies the order of the matrix A. n >= 0.

    @param[in]
    dA      COMPLEX_16 array of dimension ( lda, n )
            Before entry with uplo = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    ldda    INTEGER.
            On entry, lda specifies the first dimension of A.
            lda >= max( 1, n ).

    @param[in]
    db      COMPLEX_16 array of dimension  n
            On exit, b is overwritten with the solution vector X.

    @param[in]
    incb    INTEGER.
            On entry,  incb specifies the increment for the elements of
            b. incb must not be zero.
            Unchanged on exit.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trsv
*******************************************************************************/
extern "C" void
magmablas_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr db, magma_int_t incb,
    magma_queue_t queue)
{
    magma_int_t size_x = n * incb;

    magmaDoubleComplex_ptr dx=NULL;

    magma_zmalloc( &dx, size_x );

    magmablas_zlaset( MagmaFull, n, 1, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dx, n, queue );

    magmablas_ztrsv_recursive_outofplace( uplo, trans, diag, n, dA, ldda, db, incb, dx, queue );

    magmablas_zlacpy( MagmaFull, n, 1, dx, n, db, n, queue );

    magma_queue_sync( queue );
    magma_free( dx );
}
