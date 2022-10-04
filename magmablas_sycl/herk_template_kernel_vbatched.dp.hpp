/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/
#ifndef HERK_TEMPLATE_KERNEL_VBATCHED_CUH
#define HERK_TEMPLATE_KERNEL_VBATCHED_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp"
#include "gemm_template_device.dp.hpp"

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void herk_template_vbatched_nt_kernel(
    magma_uplo_t uplo, magma_int_t* N, magma_int_t* K,
    T alpha,
    T const * const * Aarray, magma_int_t* LDA,
    T const * const * Barray, magma_int_t* LDB,
    T beta, T**       Carray, magma_int_t* LDC, sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_N = (int)N[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ((uplo == MagmaLower) &&
        (item_ct1.get_group(1) * BLK_N > (item_ct1.get_group(2) + 1) * BLK_M))
        return;

    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ((uplo == MagmaUpper) &&
        (item_ct1.get_group(2) * BLK_M > (item_ct1.get_group(1) + 1) * BLK_N))
        return;

    gemm_template_device_nt<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_N, my_N, (int)K[batchid], Aarray[batchid], (int)LDA[batchid],
        Barray[batchid], (int)LDB[batchid], Carray[batchid], (int)LDC[batchid],
        alpha, beta, item_ct1, sA, sB);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static 
void herk_template_vbatched_tn_kernel(
    magma_uplo_t uplo, magma_int_t* N, magma_int_t* K,
    T alpha, T const * const * Aarray, magma_int_t* LDA,
    T const * const * Barray, magma_int_t* LDB,
    T beta, T**       Carray, magma_int_t* LDC , sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
    const int batchid = item_ct1.get_group(0);
    const int my_N = (int)N[batchid];
    if (item_ct1.get_group(2) >= magma_ceildiv(my_N, BLK_M)) return;
    if (item_ct1.get_group(1) >= magma_ceildiv(my_N, BLK_N)) return;

    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ((uplo == MagmaLower) &&
        (item_ct1.get_group(1) * BLK_N > (item_ct1.get_group(2) + 1) * BLK_M))
        return;

    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ((uplo == MagmaUpper) &&
        (item_ct1.get_group(2) * BLK_M > (item_ct1.get_group(1) + 1) * BLK_N))
        return;

    gemm_template_device_tn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X),
                            (BLK_N / DIM_Y), CONJA, CONJB>(
        my_N, my_N, (int)K[batchid], Aarray[batchid], (int)LDA[batchid],
        Barray[batchid], (int)LDB[batchid], Carray[batchid], (int)LDC[batchid],
        alpha, beta, item_ct1, sA, sB);
}


/******************************************************************************/
// kernel wrappers
// NT, NC
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void herk_template_vbatched_nt(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t* k,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dB_array, magma_int_t* lddb,
    T**       dC_array, magma_int_t* lddc,
    T alpha, T beta,
    magma_int_t batchCount, magma_queue_t queue,
    magma_int_t max_n)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_n, BLK_M));
        /*
        DPCT1049:153: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 2, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<2>(BLK_K, BLK_M + 1), cgh);
                sycl::accessor<T, 2, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<2>(BLK_N, BLK_K + 1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        herk_template_vbatched_nt_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            uplo, n + i, k + i, alpha, dA_array + i, ldda + i,
                            dB_array + i, lddb + i, beta, dC_array + i,
                            lddc + i, item_ct1, sA_acc_ct1, sB_acc_ct1);
                    });
            });
    }
}


/******************************************************************************/
// TN, CN
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void herk_template_vbatched_tn(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t* k,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dB_array, magma_int_t* lddb,
    T**       dC_array, magma_int_t* lddc,
    T alpha, T beta,
    magma_int_t batchCount, magma_queue_t queue,
    magma_int_t max_n)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> dimBlock(1, DIM_Y, DIM_X);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> dimGrid(ibatch, magma_ceildiv(max_n, BLK_N),
                               magma_ceildiv(max_n, BLK_M));
        /*
        DPCT1049:154: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 2, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<2>(BLK_K, BLK_M + 1), cgh);
                sycl::accessor<T, 2, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<2>(BLK_N, BLK_K + 1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        herk_template_vbatched_tn_kernel<
                            T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA,
                            DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>(
                            uplo, n + i, k + i, alpha, dA_array + i, ldda + i,
                            dB_array + i, lddb + i, beta, dC_array + i,
                            lddc + i, item_ct1, sA_acc_ct1, sB_acc_ct1);
                    });
            });
    }
}

#endif //HERK_TEMPLATE_KERNEL_VBATCHED_CUH
