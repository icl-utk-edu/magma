/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       NOTE: There is a likely compiler bug affecting this file, specifically
         the generated file in single precision (sgetrf). See below in the file
         for an explanation.

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.dp.hpp"
#include "shuffle.dp.hpp"
#include "batched_kernel_param.h"

// use this so magmasubs will replace with relevant precision, so we can comment out
// the switch case that causes compilation failure
#define PRECISION_z

#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)             (max(1,64/M))
#endif

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap.
template<int N, int NPOW2>

#ifdef MAGMA_HAVE_HIP
__launch_bounds__(NTCOL(N)*NPOW2)
#endif
void
zgetrf_batched_smallsq_noshfl_kernel( magmaDoubleComplex** dA_array, int ldda,
                                magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount,
                                sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int batchid =
        item_ct1.get_group(2) * item_ct1.get_local_range(1) + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];

    /*
    DPCT1064:733: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    magmaDoubleComplex reg    = MAGMA_Z_ZERO;
    magmaDoubleComplex update = MAGMA_Z_ZERO;

    int max_id, rowid = tx;
    int linfo = 0;
    double rx_abs_max = MAGMA_D_ZERO;

    magmaDoubleComplex *sx = (magmaDoubleComplex*)(zdata);
    double *dsx = (double *)(sx + item_ct1.get_local_range(1) * NPOW2);
    int *sipiv = (int *)(dsx + item_ct1.get_local_range(1) * NPOW2);
    sx    += ty * NPOW2;
    dsx   += ty * NPOW2;
    sipiv += ty * NPOW2;

    // read
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        // izamax and find pivot
        dsx[rowid] = sycl::fabs(MAGMA_Z_REAL(rA[i])) + sycl::fabs(MAGMA_Z_IMAG(rA[i]));
        magmablas_syncwarp(item_ct1);
        rx_abs_max = dsx[i];
        max_id = i;
        #pragma unroll
        for(int j = i+1; j < N; j++){
            if( dsx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = dsx[j];
            }
        }
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        /*
        DPCT1064:734: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        update = (rx_abs_max == MAGMA_D_ZERO) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = i; j < N; j++){
                sx[j] = update * rA[j];
            }
        }
        else if(rowid == i){
            rowid = max_id;
        }
        magmablas_syncwarp(item_ct1);

        /*
        DPCT1064:735: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        reg = (rx_abs_max == MAGMA_D_ZERO) ? MAGMA_Z_ONE
                                           : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i]);
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        magmablas_syncwarp(item_ct1);
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }
    // write
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);    // fortran indexing
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
magma_zgetrf_batched_smallsq_noshfl(
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

    const magma_int_t ntcol = 1; //magma_get_zgetrf_batched_ntcol(m, n);
				 // Set to 1 for SYCL for now to avoid early return/barrier issue
    /*
    DPCT1083:737: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    magma_int_t shmem = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(double);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(magmaDoubleComplex);
    sycl::range<3> threads(1, ntcol, magma_ceilpow2(m));
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, gridx);

    /* @author: Cade Brown <cbrow216@vols.utk.edu>
     * @date  : 2020-01-31
     *
     * Something very odd is happening with this file. The file never finishes compiling,
     * causing compilation to hang indefinitely. I've only see it apply to  It is likely a bug in either:
     *   * clang/clang++ compiler (C++ templating). I think it may be hanging on an invalid template parameter
     *       or searching through template matches in an infinite loop
     *   * LLVM code generation (specifically, the AMDGPU backend, as it seems that the compilation crashes
     *       during code generation in LL IR).
     *
     * I've only observed this when the file `magmablas_hip/sgetrf_batched_smallsq_noshfl.hip.cpp` is generated,
     * never zgetrf or other precisions.
     *
     */

    switch(m){
        /*
        DPCT1049:736: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<1, magma_ceilpow2(1)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:738: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<2, magma_ceilpow2(2)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:739: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<3, magma_ceilpow2(3)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:740: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<4, magma_ceilpow2(4)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:741: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<5, magma_ceilpow2(5)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:742: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<6, magma_ceilpow2(6)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:743: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<7, magma_ceilpow2(7)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:744: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<8, magma_ceilpow2(8)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:745: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<9, magma_ceilpow2(9)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:746: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<10,
                                                             magma_ceilpow2(10)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:747: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<11,
                                                             magma_ceilpow2(11)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;

// here are the offending cases
        /*
        DPCT1049:748: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<12,
                                                             magma_ceilpow2(12)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:749: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<13,
                                                             magma_ceilpow2(13)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:750: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<14,
                                                             magma_ceilpow2(14)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:751: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<15,
                                                             magma_ceilpow2(15)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:752: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<16,
                                                             magma_ceilpow2(16)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:753: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<17,
                                                             magma_ceilpow2(17)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:754: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<18,
                                                             magma_ceilpow2(18)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:755: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<19,
                                                             magma_ceilpow2(19)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:756: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<20,
                                                             magma_ceilpow2(20)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:757: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<21,
                                                             magma_ceilpow2(21)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:758: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<22,
                                                             magma_ceilpow2(22)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:759: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<23,
                                                             magma_ceilpow2(23)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:760: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<24,
                                                             magma_ceilpow2(24)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:761: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<25,
                                                             magma_ceilpow2(25)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:762: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<26,
                                                             magma_ceilpow2(26)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;

        /*
        DPCT1049:763: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<27,
                                                             magma_ceilpow2(27)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:764: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<28,
                                                             magma_ceilpow2(28)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:765: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<29,
                                                             magma_ceilpow2(29)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:766: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<30,
                                                             magma_ceilpow2(30)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:767: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<31,
                                                             magma_ceilpow2(31)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
        /*
        DPCT1049:768: The work-group size passed to the SYCL kernel may exceed
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
                        zgetrf_batched_smallsq_noshfl_kernel<32,
                                                             magma_ceilpow2(32)>(
                            dA_array, ldda, ipiv_array, info_array, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });
            break;
/**/

        // replace the default error message with something so people can contact me
        //default: printf("error: size %lld is not supported\n", (long long) m);
        default: fprintf(stderr, "MAGMA: error in *getrf_batched_smallsq_noshfl, unsupported size '%lld'. Please contact Cade Brown <cbrow216@vols.utk.edu>, or some member of the MAGMA team with details about this application.\n", (long long)m);

    }
    return arginfo;
}
