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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define    TRTRI_NONBATCHED
#include "ztrtri.dp.hpp"

/***************************************************************************//**
    Purpose
    -------
    ztrtri_diag inverts the NB x NB diagonal blocks of A.

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
    dA      COMPLEX_16 array of dimension ( ldda, n )
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
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    d_dinvA COMPLEX_16 array of dimension (NB, ceil(n/NB)*NB),
            where NB = 128.
            On exit, contains inverses of the NB-by-NB diagonal blocks of A.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trtri_diag
*******************************************************************************/
extern "C" void
magmablas_ztrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr d_dinvA,
    magma_queue_t queue)
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

    dpct::get_default_queue()
        .memset(d_dinvA, 0,
                magma_roundup(n, NB) * NB * sizeof(magmaDoubleComplex))
        .wait();

    if ( uplo == MagmaLower ) {
        // invert diagonal IB x IB inner blocks
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<1>(IB*IB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                          sycl::range<3>(1, 1, IB),
                                      sycl::range<3>(1, 1, IB)),
                    [=](sycl::nd_item<3> item_ct1) {
                        ztrtri_diag_lower_kernel(diag, n, dA, ldda, d_dinvA,
                                                 item_ct1,
                                                 sB_acc_ct1.get_pointer());
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
                1, npages * (jb / 16),
                jb / (threads[2] *
                      threads[1])); // emulate 3D grid: NX * (NY*npages), for
                                    // CUDA ARCH 1.x

            //printf( "n %d, jb %d, grid %d x %d (%d x %d)\n", n, jb, grid.x, grid.y, grid.y / npages, npages );
            switch (jb) {
                case 16:
                    /*
                    DPCT1049:1564: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm16_part1_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1559: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm16_part2_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    break;
                case 32:
                    /*
                    DPCT1049:1565: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm32_part1_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1560: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm32_part2_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    break;
                case 64:
                    /*
                    DPCT1049:1566: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm64_part1_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1561: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm64_part2_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    break;
                default:
                    /*
                    DPCT1049:1567: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm_above64_part1_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1562: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm_above64_part2_lower_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1563: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                       triple_zgemm_above64_part3_lower_kernel(
                                           n, dA, ldda, d_dinvA, jb, npages,
                                           item_ct1);
                                   });
                    break;
            }
            if ( kb >= n ) break;
        }
    }
    else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<1>(IB*IB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                          sycl::range<3>(1, 1, IB),
                                      sycl::range<3>(1, 1, IB)),
                    [=](sycl::nd_item<3> item_ct1) {
                        ztrtri_diag_upper_kernel(diag, n, dA, ldda, d_dinvA,
                                                 item_ct1,
                                                 sB_acc_ct1.get_pointer());
                    });
            });

        // update the inverse up to the size of IB
        for( int jb=IB; jb < NB; jb *= 2 ) {
            int kb = jb*2;
            int npages = magma_ceildiv( n, kb );
            sycl::range<3> threads(1, 4, (jb <= 32 ? jb / 4 : 16));
            sycl::range<3> grid(
                1, npages * (jb / 16),
                jb / (threads[2] *
                      threads[1])); // emulate 3D grid: NX * (NY*npages), for
                                    // CUDA ARCH 1.x

            switch (jb) {
                case 16:
                    /*
                    DPCT1049:1573: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm16_part1_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1568: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm16_part2_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    break;
                case 32:
                    /*
                    DPCT1049:1574: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm32_part1_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1569: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm32_part2_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    break;
                case 64:
                    /*
                    DPCT1049:1575: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm64_part1_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1570: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm64_part2_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    break;
                default:
                    /*
                    DPCT1049:1576: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm_above64_part1_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1571: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<magmaDoubleComplex, 2,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            sB_acc_ct1(sycl::range<2>(16, 17), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                triple_zgemm_above64_part2_upper_kernel(
                                    n, dA, ldda, d_dinvA, jb, npages, item_ct1,
                                    sB_acc_ct1);
                            });
                    });
                    /*
                    DPCT1049:1572: The work-group size passed to the SYCL kernel
                    may exceed the limit. To get the device limit, query
                    info::device::max_work_group_size. Adjust the work-group
                    size if needed.
                    */
                ((sycl::queue *)(queue->sycl_stream()))
                    ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                       triple_zgemm_above64_part3_upper_kernel(
                                           n, dA, ldda, d_dinvA, jb, npages,
                                           item_ct1);
                                   });
                    break;
            }
            if ( kb >= n ) break;
        }
    }
}
