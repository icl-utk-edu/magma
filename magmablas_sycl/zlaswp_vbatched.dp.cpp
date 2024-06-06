/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah
*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "zlaswp_device.dp.hpp"

#define PRECISION_z
#define ib    (3)

/******************************************************************************/
// serial swap that does swapping one row by one row
// this is the vbatched routine, for swapping to the left of the panel
void zlaswp_left_rowserial_kernel_vbatched(
                int n,
                magma_int_t *M, magma_int_t *N,
                magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
                magma_int_t** ipiv_array, int ipiv_i,
                int k1, int k2 , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int tid = item_ct1.get_local_id(2) +
                    item_ct1.get_local_range(2) * item_ct1.get_group(2);

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_minmn = min(my_M, my_N);
    int my_ldda  = (int)ldda[batchid];
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *dipiv     = ipiv_array[batchid] + ipiv_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    //k1--;
    //k2--;

    // reduce minmn by the pivot offset
    my_minmn -= ipiv_i;

    // check my_minmn & the offsets k1, k2
    if( my_minmn <= 0 || k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    // if we use k2 = min(k2, my_minmn-1),
    // then the for loop below should be until i1 <= k2, not i1 < k2

    // the following explanation is based on the assumption m >= n for all matrices
    // since this is a separate kernel for left-swap, we can calculate the maximum
    // affordable n based on (Ai, Aj).
    // In a left swap, Ai > Aj, which means (Ai, Aj) is on the left of the diagonal element
    // If the diagonal (Ai, Ai) is inside the matrix, then my_max_n is the horizontal
    // distance between (Ai, Aj) and (Ai, Ai). If (Ai, Ai) is outside a given matrix, we
    // terminate the thread-block(s) for this matrix only
    if(my_M < Ai || my_N < Ai) return;
    const int my_max_n = Ai - Aj;
    const int my_n     = min(n, my_max_n);

    if (tid < my_n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++) {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1 ) {
                A1 = dA[i1 + tid * my_ldda];
                dA[i1 + tid * my_ldda] = dA[i2 + tid * my_ldda];
                dA[i2 + tid * my_ldda] = A1;
            }
        }
    }
}

/******************************************************************************/
// serial swap that does swapping one row by one row
// this is the vbatched routine, for swapping to the right of the panel
void zlaswp_right_rowserial_kernel_vbatched(
                int n,
                magma_int_t *M, magma_int_t *N,
                magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
                magma_int_t** ipiv_array, int ipiv_i,
                int k1, int k2 , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    const int tid = item_ct1.get_local_id(2) +
                    item_ct1.get_local_range(2) * item_ct1.get_group(2);

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_minmn = min(my_M, my_N);
    int my_ldda  = (int)ldda[batchid];
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *dipiv     = ipiv_array[batchid] + ipiv_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    //k1--;
    //k2--;

    // reduce minmn by the pivot offset
    my_minmn -= ipiv_i;

    // check minmn, & the offsets k1, k2
    if( my_minmn <= 0 || k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    // if we use k2 = min(k2, my_minmn-1),
    // then the for loop below should be until i1 <= k2, not i1 < k2


    // check the input scalar 'n'
    const int my_max_n = my_N - Aj;
    const int my_n     = min(n, my_max_n);

    if (tid < my_n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++) {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1 ) {
                A1 = dA[i1 + tid * my_ldda];
                dA[i1 + tid * my_ldda] = dA[i2 + tid * my_ldda];
                dA[i2 + tid * my_ldda] = A1;
            }
        }
    }
}

/******************************************************************************/

void zlaswp_left_rowparallel_kernel_vbatched(
                                int n, int width,
                                magma_int_t* M, magma_int_t* N,
                                magmaDoubleComplex **dA_array,  int Ai,  int Aj,  magma_int_t* ldda,
                                magma_int_t** pivinfo_array, int pivinfo_i,
                                int k1, int k2, sycl::nd_item<3> item_ct1,
                                uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(0);

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_ldda  = (int)ldda[batchid];

    magmaDoubleComplex* dA = dA_array[batchid]  + Aj  * my_ldda + Ai;
    magma_int_t *pivinfo   = pivinfo_array[batchid] + pivinfo_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    //my_M -= Ai;
    //my_N -= Aj;
    int my_minmn = min(my_M, my_N);

    // reduce minmn by the pivot offset
    my_minmn -= pivinfo_i;
    if( my_minmn <= 0  ) return;
    if( k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    const int my_height = k2-k1;

    // the following explanation is based on the assumption m >= n for all matrices
    // since this is a separate kernel for left-swap, we can calculate the maximum
    // affordable n based on (Ai, Aj).
    // In a left swap, Ai > Aj, which means (Ai, Aj) is on the left of the diagonal element
    // If the diagonal (Ai, Ai) is inside the matrix, then my_max_n is the horizontal
    // distance between (Ai, Aj) and (Ai, Ai). If (Ai, Ai) is outside a given matrix, we
    // terminate the thread-block(s) for this matrix only
    if(my_M < Ai || my_N < Ai) return;
    const int my_max_n = Ai - Aj;
    const int my_n     = min(n, my_max_n);
    zlaswp_rowparallel_devfunc(my_n, width, my_height, dA, my_ldda, dA, my_ldda,
                               pivinfo, item_ct1, dpct_local);
}

/******************************************************************************/
// serial swap that does swapping one row by one row
// this is the vbatched routine, for swapping to the right of the panel
void zlaswp_right_rowparallel_kernel_vbatched(
                int n, int width,
                magma_int_t *M, magma_int_t *N,
                magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
                magma_int_t** pivinfo_array, int pivinfo_i,
                int k1, int k2 , sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    const int batchid = item_ct1.get_group(0);

    int my_M     = (int)M[batchid];
    int my_N     = (int)N[batchid];
    int my_minmn = min(my_M, my_N);
    int my_ldda  = (int)ldda[batchid];
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *dipivinfo = pivinfo_array[batchid] + pivinfo_i;

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    // reduce minmn by the pivot offset
    my_minmn -= pivinfo_i;

    // check minmn, & the offsets k1, k2
    if( my_minmn <= 0 || k1 >= my_minmn ) return;
    k2 = min(k2, my_minmn);
    const int my_height = k2-k1;

    // check the input scalar 'n'
    const int my_max_n = my_N - Aj;
    const int my_n     = min(n, my_max_n);

    zlaswp_rowparallel_devfunc(my_n, width, my_height, dA, my_ldda, dA, my_ldda,
                               dipivinfo, item_ct1, dpct_local);
}


/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_left_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t blocks         = magma_ceildiv( n, BLK_SIZE );
    magma_int_t min_BLK_SIZE_n = min(BLK_SIZE, n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        /*
        DPCT1049:1275: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, min_BLK_SIZE_n),
                                  sycl::range<3>(1, 1, min_BLK_SIZE_n)),
                [=](sycl::nd_item<3> item_ct1) {
                    zlaswp_left_rowserial_kernel_vbatched(
                        n, M, N, dA_array, Ai, Aj, ldda, ipiv_array,
                        ipiv_offset, k1, k2, item_ct1);
                });
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_right_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t blocks         = magma_ceildiv( n, BLK_SIZE );
    magma_int_t min_BLK_SIZE_n = min(BLK_SIZE, n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        /*
        DPCT1049:1276: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, min_BLK_SIZE_n),
                                  sycl::range<3>(1, 1, min_BLK_SIZE_n)),
                [=](sycl::nd_item<3> item_ct1) {
                    zlaswp_right_rowserial_kernel_vbatched(
                        n, M, N, dA_array, Ai, Aj, ldda, ipiv_array,
                        ipiv_offset, k1, k2, item_ct1);
                });
    }
}

/******************************************************************************/
extern "C" void
magma_zlaswp_left_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024) {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t width = min(n, SWP_WIDTH);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        /*
        DPCT1083:1278: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shmem = sizeof(magmaDoubleComplex) * height * width;
        /*
        DPCT1049:1277: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, height),
                                      sycl::range<3>(1, 1, height)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zlaswp_left_rowparallel_kernel_vbatched(
                            n, width, M, N, dA_array, Ai, Aj, ldda,
                            pivinfo_array, pivinfo_i, k1, k2, item_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
}

/******************************************************************************/
extern "C" void
magma_zlaswp_right_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024) {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();
    magma_int_t width = min(n, SWP_WIDTH);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, blocks);

        /*
        DPCT1083:1280: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shmem = sizeof(magmaDoubleComplex) * height * width;
        /*
        DPCT1049:1279: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, height),
                                      sycl::range<3>(1, 1, height)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zlaswp_right_rowparallel_kernel_vbatched(
                            n, width, M, N, dA_array, Ai, Aj, ldda,
                            pivinfo_array, pivinfo_i, k1, k2, item_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
}

