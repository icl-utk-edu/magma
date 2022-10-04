/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef HEMV_TEMPLATE_KERNEL_VBATCHED_CUH
#define HEMV_TEMPLATE_KERNEL_VBATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "atomics.dp.hpp"
#include "hemv_template_device.dp.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static 
void hemv_diag_template_vbatched_kernel(
        magma_uplo_t uplo, magma_int_t* n,
        T alpha, T** Aarray, magma_int_t* ldda,
                 T** Xarray, magma_int_t* incx,
        T beta,  T** Yarray, magma_int_t* incy,
        int max_N,
        int offA, int offX, int offY,
        int spec_N, sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    const int batchid = item_ct1.get_group(0);
    int my_N = (int)n[batchid];
    // check if the offset produces an out-of-bound pointer
    if( my_N < offA) return;
    // compute the maximum allowed n
    my_N -= offA;
    // check if the user forces n
    my_N = ( spec_N <= 0 ) ? my_N : min( my_N, spec_N );

    if( my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Xarray[batchid] == NULL || Yarray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, NB)) return;
    hemv_diag_device<T, NB, TY>(
        uplo, my_N, alpha, Aarray[batchid] + offA * (int)ldda[batchid] + offA,
        (int)ldda[batchid], Xarray[batchid] + offX * (int)incx[batchid],
        (int)incx[batchid], beta, Yarray[batchid] + offY * (int)incy[batchid],
        (int)incy[batchid], item_ct1, sA, sX);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static 
void hemv_lower_template_vbatched_kernel(
        magma_int_t* n, T alpha,
        T** Aarray, magma_int_t* ldda,
        T** Xarray, magma_int_t* incx,
        T** Yarray, magma_int_t* incy,
        int max_N,
        int offA, int offX, int offY,
        int spec_N, sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    const int batchid = item_ct1.get_group(0);
    int my_N = (int)n[batchid];
    // check if the offset produces an out-of-bound pointer
    if( my_N < offA) return;
    // compute the maximum allowed n
    my_N -= offA;
    // check if the user forces n
    my_N = ( spec_N <= 0 ) ? my_N : min( my_N, spec_N );

    if( my_N <= NB ) return;    // sizes <= NB are handled by the diag kernel
    if( Aarray[batchid] == NULL || Xarray[batchid] == NULL || Yarray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, NB)) return;
    hemv_lower_device<T, NB, TY>(
        my_N, alpha, Aarray[batchid] + offA * (int)ldda[batchid] + offA,
        (int)ldda[batchid], Xarray[batchid] + offX * (int)incx[batchid],
        (int)incx[batchid], Yarray[batchid] + offY * (int)incy[batchid],
        (int)incy[batchid], item_ct1, sA, sX);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
static 
void hemv_upper_template_vbatched_kernel(
        magma_int_t* n, T alpha,
        T** Aarray, magma_int_t* ldda,
        T** Xarray, magma_int_t* incx,
        T** Yarray, magma_int_t* incy,
        int max_N,
        int offA, int offX, int offY,
        int spec_N, sycl::nd_item<3> item_ct1, T *sA, T *sX)
{
    const int batchid = item_ct1.get_group(0);
    int my_N = (int)n[batchid];
    // check if the offset produces an out-of-bound pointer
    if( my_N < offA) return;
    // compute the maximum allowed n
    my_N -= offA;
    // check if the user forces n
    my_N = ( spec_N <= 0 ) ? my_N : min( my_N, spec_N );

    if( my_N <= NB ) return;    // sizes <= NB are handled by the diag kernel
    if( Aarray[batchid] == NULL || Xarray[batchid] == NULL || Yarray[batchid] == NULL ) return;
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, NB)) return;
    hemv_upper_device<T, NB, TY>(
        my_N, alpha, Aarray[batchid] + offA * (int)ldda[batchid] + offA,
        (int)ldda[batchid], Xarray[batchid] + offX * (int)incx[batchid],
        (int)incx[batchid], Yarray[batchid] + offY * (int)incy[batchid],
        (int)incy[batchid], item_ct1, sA, sX);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// diag
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_diag_template_vbatched(
    magma_uplo_t uplo, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t* ldda,
             T** dX_array, magma_int_t* incx,
    T beta,  T** dY_array, magma_int_t* incy,
    magma_int_t max_n,
    magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t spec_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, TY, NB);

    for(magma_int_t i=0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_n, NB));

        /*
        DPCT1049:1027: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * NB), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sX_acc_ct1(sycl::range<1>(NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        hemv_diag_template_vbatched_kernel<T, NB, TY>(
                            uplo, n + i, alpha, dA_array + i, ldda + i,
                            dX_array + i, incx + i, beta, dY_array + i,
                            incy + i, max_n, offA, offX, offY, spec_n, item_ct1,
                            sA_acc_ct1.get_pointer(), sX_acc_ct1.get_pointer());
                    });
            });
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lower
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_lower_template_vbatched(
    magma_int_t* n, T alpha,
    T** dA_array, magma_int_t* ldda,
    T** dX_array, magma_int_t* incx,
    T** dY_array, magma_int_t* incy,
    magma_int_t max_n,
    magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t spec_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, TY, NB);

    for(magma_int_t i=0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_n, NB));

        /*
        DPCT1049:1028: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * (NB + 1)), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sX_acc_ct1(sycl::range<1>(NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        hemv_lower_template_vbatched_kernel<T, NB, TY>(
                            n + i, alpha, dA_array + i, ldda + i, dX_array + i,
                            incx + i, dY_array + i, incy + i, max_n, offA, offX,
                            offY, spec_n, item_ct1, sA_acc_ct1.get_pointer(),
                            sX_acc_ct1.get_pointer());
                    });
            });
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// upper
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int TY>
void hemv_upper_template_vbatched(
    magma_int_t* n, T alpha,
    T** dA_array, magma_int_t* ldda,
    T** dX_array, magma_int_t* incx,
    T** dY_array, magma_int_t* incy,
    magma_int_t max_n,
    magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t spec_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, TY, NB);

    for(magma_int_t i=0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_n, NB));

        /*
        DPCT1049:1029: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<1>(NB * (NB + 1)), cgh);
                sycl::accessor<T, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sX_acc_ct1(sycl::range<1>(NB), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        hemv_upper_template_vbatched_kernel<T, NB, TY>(
                            n + i, alpha, dA_array + i, ldda + i, dX_array + i,
                            incx + i, dY_array + i, incy + i, max_n, offA, offX,
                            offY, spec_n, item_ct1, sA_acc_ct1.get_pointer(),
                            sX_acc_ct1.get_pointer());
                    });
            });
    }
}
#endif //HEMV_TEMPLATE_KERNEL_VBATCHED_CUH
