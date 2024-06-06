/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar

       File named ztrtri_diag.cu to avoid name conflict with src/ztrtri.o
       in the library. The actual kernels are in ztrtri_lower.cu and ztrtri_upper.cu
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define TRTRI_BATCHED
#include "ztrtri.dp.hpp"

/***************************************************************************//**
    Purpose
    -------
    ZTRTRI_DIAG inverts the NB x NB diagonal blocks of a triangular matrix.
    This routine is used in ztrsm.

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n specifies the order of the matrix A. N >= 0.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A of dimension ( ldda, n )
             The triangular matrix A.
    \n
             If UPLO = MagmaUpper, the leading N-by-N upper triangular part of A
             contains the upper triangular matrix, and the strictly lower
             triangular part of A is not referenced.
    \n
            If UPLO = MagmaLower, the leading N-by-N lower triangular part of A
            contains the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.
    \n
            If DIAG = MagmaUnit, the diagonal elements of A are also not referenced
            and are assumed to be 1.

    @param[in]
    ldda    INTEGER.
            The leading dimension of each array A.  LDDA >= max(1,N).

    @param[out]
    dinvA_array Array of pointers, dimension (batchCount).
                Each is a COMPLEX_16 array dinvA of dimension (NB, ceil(n/NB)*NB),
                where NB = 128.
                On exit, contains inverses of the NB-by-NB diagonal blocks of A.

    @param[in]
    resetozero INTEGER
               If not zero, each array dinvA will be reset to all zeros

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trtri_diag_batched
*******************************************************************************/
extern "C" void
magmablas_ztrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex const * const *dA_array, magma_int_t ldda,
    magmaDoubleComplex **dinvA_array,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if (uplo != MagmaLower && uplo != MagmaUpper)
        info = -1;
    else if (diag != MagmaNonUnit && diag != MagmaUnit)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (ldda < n)
        info = -5;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info
    }

    int nblocks = magma_ceildiv( n, IB );

    if ( resetozero ) {
        /*
        DPCT1064:1540: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmablas_zlaset_batched(MagmaFull, magma_roundup(n, NB), NB,
                                 MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA_array,
                                 magma_roundup(n, NB), batchCount, queue);
    }
    // if someone want to use cudamemset he need to set the whole vectors
    // of initial size otherwise it is a bug and thus need to have dinvA_length
    // in input parameter and has been tested and was slower.
    //was not the largest size computed by the high API getrf_batched then it is bug and need to use magmablas_zlaset_batched

    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);

        if ( uplo == MagmaLower ) {
            // invert diagonal IB x IB inner blocks
            sycl::range<3> diaggrid(ibatch, 1, nblocks); // emulate 3D grid
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        sB_acc_ct1(sycl::range<1>(IB*IB), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(diaggrid * sycl::range<3>(1, 1, IB),
                                          sycl::range<3>(1, 1, IB)),
                        [=](sycl::nd_item<3> item_ct1) {
                            ztrtri_diag_lower_kernel_batched(
                                diag, n, dA_array + i, ldda, dinvA_array + i,
                                item_ct1, sB_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });

            // build up NB x NB blocks (assuming IB=16 here):
            // use   16 x 16  blocks to build  32 x 32  blocks,  1 x (1 x npages) grid,  4 x 4 threads;
            // then  32 x 32  blocks to build  64 x 64  blocks,  1 x (2 x npages) grid,  8 x 4 threads;
            // then  64 x 64  blocks to build 128 x 128 blocks,  1 x (4 x npages) grid, 16 x 4 threads;
            // then 128 x 128 blocks to build 256 x 256 blocks,  2 x (8 x npages) grid, 16 x 4 threads.
            for( int jb=IB; jb < NB; jb *= 2 ) {
                int kb = jb*2;
                int npages = magma_ceildiv( n, kb );
                sycl::range<3> threads(1, 4, (jb <= 32 ? jb / 4 : 16));
                sycl::range<3> grid(
                    ibatch, npages * (jb / 16),
                    jb / (threads[2] *
                          threads[1])); // emulate 3D grid: NX * (NY*npages),
                                        // for CUDA ARCH 1.x

                //printf( "n %d, jb %d, grid %d x %d (%d x %d)\n", n, jb, grid.x, grid.y, grid.y / npages, npages );
                switch (jb) {
                    case 16:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm16_part1_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm16_part2_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                        break;
                    case 32:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm32_part1_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm32_part2_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                        break;
                    case 64:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm64_part1_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm64_part2_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                        break;
                    default:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm_above64_part1_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm_above64_part2_lower_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm_above64_part3_lower_kernel_batched(
                                    n, dA_array + i, ldda, dinvA_array + i, jb,
                                    npages, item_ct1);
                            });
                        break;
                }
                if ( kb >= n ) break;
            }
        }
        else {
            sycl::range<3> diaggrid(ibatch, 1, nblocks); // emulate 3D grid
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1>
                        sB_acc_ct1(sycl::range<1>(IB*IB), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(diaggrid * sycl::range<3>(1, 1, IB),
                                          sycl::range<3>(1, 1, IB)),
                        [=](sycl::nd_item<3> item_ct1) {
                            ztrtri_diag_upper_kernel_batched(
                                diag, n, dA_array + i, ldda, dinvA_array + i,
                                item_ct1, sB_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });

            // update the inverse up to the size of IB
            for( int jb=IB; jb < NB; jb *= 2 ) {
                int kb = jb*2;
                int npages = magma_ceildiv( n, kb );
                sycl::range<3> threads(1, 4, (jb <= 32 ? jb / 4 : 16));
                sycl::range<3> grid(
                    ibatch, npages * (jb / 16),
                    jb / (threads[2] *
                          threads[1])); // emulate 3D grid: NX * (NY*npages),
                                        // for CUDA ARCH 1.x

                switch (jb) {
                    case 16:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm16_part1_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm16_part2_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                        break;
                    case 32:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm32_part1_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm32_part2_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                        break;
                    case 64:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm64_part1_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm64_part2_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                        break;
                    default:
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm_above64_part1_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<magmaDoubleComplex, 2>
                                sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(grid * threads, threads),
                                [=](sycl::nd_item<3> item_ct1) {
                                    triple_zgemm_above64_part2_upper_kernel_batched(
                                        n, dA_array + i, ldda, dinvA_array + i,
                                        jb, npages, item_ct1, sB_acc_ct1);
                                });
                        });
                    ((sycl::queue *)(queue->sycl_stream()))
                        ->parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm_above64_part3_upper_kernel_batched(
                                    n, dA_array + i, ldda, dinvA_array + i, jb,
                                    npages, item_ct1);
                            });
                        break;
                }
                if ( kb >= n ) break;
            }
        }
    }
}
