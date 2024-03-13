/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.dp.hpp"
#include "shuffle.dp.hpp"
#include "batched_kernel_param.h"

// This kernel uses registers for matrix storage, shared mem. and shuffle for communication.
// It also uses lazy swap.
//extern __shared__ double ddata[];
template<int N, int NSHFL>


void
zgetrf_batched_smallsq_shfl_kernel( magmaDoubleComplex** dA_array, int ldda,
                                magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount,
                                sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto ddata = (double *)dpct_local;

    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int batchid =
        item_ct1.get_group(2) * item_ct1.get_local_range(1) + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];

    /*
    DPCT1064:769: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    /*
    DPCT1064:770: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex y[N] = {MAGMA_Z_ZERO};
    magmaDoubleComplex reg    = MAGMA_Z_ZERO;
    magmaDoubleComplex update = MAGMA_Z_ZERO;

    int max_id, current_piv_tx, rowid = tx, linfo = 0;
    double rx_abs_max = MAGMA_D_ZERO;
    // shared memory pointers
    double* sx = (double*)(ddata);
    int *sipiv = (int *)(sx + item_ct1.get_local_range(1) * NSHFL);
    sx += ty * NSHFL;
    sipiv += ty * (NSHFL+1);
    volatile int* scurrent_piv_tx = (volatile int*)(sipiv + NSHFL);

    // read
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        sx[rowid] = sycl::fabs(MAGMA_Z_REAL(rA[i])) + sycl::fabs(MAGMA_Z_IMAG(rA[i]));
        magmablas_syncwarp(item_ct1);
        rx_abs_max = sx[i];
        max_id = i;
        #pragma unroll
        for(int j = i; j < N; j++){
            if( sx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = sx[j];
            }
        }
        linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        /*
        DPCT1064:771: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        update = (rx_abs_max == MAGMA_D_ZERO) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            (*scurrent_piv_tx) = tx;
        }
        else if(rowid == i){
            rowid = max_id;
        }
        current_piv_tx = (*scurrent_piv_tx);
        magmablas_syncwarp(item_ct1);

#pragma unroll
        for(int j = i; j < N; j++){
            y[j] = update *
                   magmablas_zshfl(rA[j], current_piv_tx, item_ct1, NSHFL);
        }
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, y[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * y[j];
            }
        }
    }

    // write
    if( tx == 0 ){
        (*info) = (magma_int_t)linfo;
    }
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    zgetrf_batched_smallsq_noshfl computes the LU factorization of a square N-by-N matrix A
    using partial pivoting with row interchanges.
    This routine can deal only with square matrices of size up to 32

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_batched_smallsq_shfl(
    magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m = n;

    if( (m < 0) || ( m > 32 ) ){
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0) return 0;

    const magma_int_t ntcol = 1; // magma_get_zgetrf_batched_ntcol(m, n);
				 // Set to 1 for SYCL for now to avoid early return/barrier issue
    /*
    DPCT1083:773: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    magma_int_t shmem = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(double);
                shmem += ntcol * 1 * sizeof(int);
    sycl::range<3> threads(1, ntcol, magma_ceilpow2(m));
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, gridx);
    switch(m){
        case 1: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<1, magma_ceilpow2(1)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 2: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<2, magma_ceilpow2(2)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 3: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<3, magma_ceilpow2(3)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 4: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<4, magma_ceilpow2(4)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 5: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<5, magma_ceilpow2(5)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 6: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<6, magma_ceilpow2(6)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 7: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<7, magma_ceilpow2(7)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<8, magma_ceilpow2(8)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 9: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<9, magma_ceilpow2(9)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 10: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<10, magma_ceilpow2(10)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 11: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<11, magma_ceilpow2(11)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 12: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<12, magma_ceilpow2(12)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 13: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<13, magma_ceilpow2(13)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 14: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<14, magma_ceilpow2(14)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 15: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<15, magma_ceilpow2(15)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 16: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<16, magma_ceilpow2(16)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 17: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<17, magma_ceilpow2(17)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 18: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<18, magma_ceilpow2(18)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 19: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<19, magma_ceilpow2(19)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 20: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<20, magma_ceilpow2(20)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 21: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<21, magma_ceilpow2(21)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 22: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<22, magma_ceilpow2(22)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 23: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<23, magma_ceilpow2(23)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 24: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<24, magma_ceilpow2(24)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 25: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<25, magma_ceilpow2(25)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 26: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<26, magma_ceilpow2(26)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 27: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<27, magma_ceilpow2(27)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 28: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<28, magma_ceilpow2(28)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 29: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<29, magma_ceilpow2(29)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 30: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<30, magma_ceilpow2(30)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 31: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<31, magma_ceilpow2(31)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        case 32: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                    32)]] {
                        zgetrf_batched_smallsq_shfl_kernel<32, magma_ceilpow2(32)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        default: printf("error: size %lld is not supported\n", (long long) m);
    }
    return arginfo;
}
