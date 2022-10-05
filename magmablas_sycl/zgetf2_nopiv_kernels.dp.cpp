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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.dp.hpp"
#include "shuffle.dp.hpp"
#include "batched_kernel_param.h"

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap.
//extern __shared__ magmaDoubleComplex zdata[];

template<int N>
void
zgetf2_nopiv_device(int m, magmaDoubleComplex* dA, int ldda, magma_int_t *info, const int tx, magmaDoubleComplex* sx, int gbstep,
                    sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1064:695: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    /*
    DPCT1064:696: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex reg = MAGMA_Z_ZERO;

    int linfo = 0;
    double abs;
    // check from previous calls if the panel factorization failed previously
    // this is necessary to report the correct info value 
    if(gbstep > 0 && *info != 0) return;

    // read 
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }
        
    #pragma unroll
    for(int i = 0; i < N; i++){
        if(tx == i){
            #pragma unroll
            for(int j = 0; j < N; j++)
                sx[j] = rA[j];
        }
        /*
        DPCT1065:697: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        abs = sycl::fabs(x()(sx[i])) + sycl::fabs(y()(sx[i]));
        linfo = ( abs == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;
        //linfo = ( abs  == MAGMA_D_ZERO ) ? min(linfo,gbstep+i+1):0;
        reg   = (linfo == 0 ) ? MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] ) : MAGMA_Z_ONE;

        // scal and ger
        if( tx > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        /*
        DPCT1065:698: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write
    #pragma unroll
    for(int i = 0; i < N; i++){
        dA[ i * ldda + tx ] = rA[i];
    }
}

/******************************************************************************/
template<int N, int NPOW2>
void
zgetf2_nopiv_batched_kernel( int m, magmaDoubleComplex** dA_array, int ai, int aj, int ldda, 
                             magma_int_t* info_array, int gbstep, int batchCount,
                             sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;

    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int batchid =
        item_ct1.get_group(2) * item_ct1.get_local_range(1) + ty;
    if(batchid >= batchCount)return;

    magmaDoubleComplex* dA = dA_array[batchid] + aj * ldda + ai;
    magma_int_t* info = &info_array[batchid];
    magmaDoubleComplex* sx = (magmaDoubleComplex*)zdata;
    sx += ty * NPOW2;

    zgetf2_nopiv_device<N>(m, dA, ldda, info, tx, sx, gbstep, item_ct1);
}
/***************************************************************************//**
    Purpose
    -------
    zgetf2_nopiv computes the non-pivoting LU factorization of an M-by-N matrix A.
    This routine can deal with matrices of limited widths, so it is for internal use.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is a batched version that factors batchCount M-by-N matrices in parallel.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows the matrix A.  N >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for dA_array.

    @param[in]
    aj      INTEGER
            Column offset for dA_array.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

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
    gbstep      INTEGER
                Internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t 
magma_zgetf2_nopiv_internal_batched( 
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    magma_int_t* info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue )
{
    #define dAarray(i,j) dA_array, i, j

    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 || n > 32 || (m > 512 && n > 16) ) {
        arginfo = -2;
    } else if (ai < 0) {
        arginfo = -4;
    } else if (aj < 0) {
        arginfo = -5;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t m1 = (m > MAX_NTHREADS) ? MAX_NTHREADS : m;
    magma_int_t m2 = m - m1;

    const magma_int_t ntcol = (m1 > 32) ? 1 : (2 * (32/m1));
    /*
    DPCT1083:700: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    magma_int_t shmem = ntcol * magma_ceilpow2(n) * sizeof(magmaDoubleComplex);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> threads(1, ntcol, m1);
    sycl::range<3> grid(1, 1, gridx);
    switch(n){
        /*
        DPCT1049:699: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 1: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<1, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:701: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 2: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<2, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:702: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 3: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<3, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:703: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 4: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<4, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:704: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 5: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<5, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:705: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 6: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<6, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:706: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 7: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<7, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:707: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<8, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:708: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 9: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<9, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:709: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 10: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<10, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:710: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 11: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<11, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:711: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 12: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<12, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:712: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 13: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<13, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:713: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 14: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<14, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:714: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 15: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<15, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:715: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 16: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<16, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:716: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 17: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<17, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:717: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 18: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<18, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:718: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 19: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<19, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:719: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 20: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<20, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:720: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 21: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<21, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:721: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 22: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<22, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:722: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 23: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<23, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:723: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 24: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<24, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:724: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 25: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<25, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:725: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 26: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<26, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:726: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 27: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<27, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:727: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 28: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<28, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:728: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 29: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<29, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:729: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 30: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<30, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:730: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 31: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<31, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:731: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 32: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_batched_kernel<32, magma_ceilpow2>(
                            m1, dA_array, ai, aj, ldda, info_array, gbstep,
                            batchCount, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        default: printf("error: panel width %lld is not supported\n", (long long) n);
    }

    if(m2 > 0){
        magmablas_ztrsm_recursive_batched(
            MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
            /*
            DPCT1064:732: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            m2, n, MAGMA_Z_ONE, dAarray(ai, aj), ldda, dAarray(ai + m1, aj),
            ldda, batchCount, queue);
    }

    #undef dAarray
    return arginfo;
}
