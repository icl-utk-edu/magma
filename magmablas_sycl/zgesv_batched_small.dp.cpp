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

#ifdef PRECISION_z
#define MAX_N    (53)
#else
#define MAX_N    (60)
#endif

#define SLDA(n)  ( (n == 7 || n == 15 || n == 23 || n == 31) ? (n) : (n+1) )
#define sA(i,j)  sA[(j)*slda + (i)]
#define sB(i,j)  sB[(j)*sldb + (i)]

// tx    : thread ID in the x dimension
// rA[N] : register array holding the A matrix
// sipiv : shared memory workspace, size N, holds the pivot vector on exits
// rB    : scalar holding the right hand side on entry (one element per thread)
// sB    : shared memory workspace, size N, holds the solution of Ax=B on exit
// sx    : shared memory workspace, size N, needed internally
// dsx   : shared memory workspace, size N, needed internally
// rowid : integer scalar, represents the row interchanges as a result of partial pivoting
// linfo : info output (non-zero means an error has occurred)
template<int N>
__inline__ void
zgesv_batched_small_device(
    const int tx,
    magmaDoubleComplex rA[N], int* sipiv,
    magmaDoubleComplex &rB, magmaDoubleComplex *sB,
    magmaDoubleComplex *sx, double *dsx,
    int &rowid, int &linfo , sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1064:508: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex reg = MAGMA_Z_ZERO;
    int max_id;

    #pragma unroll
    for(int i = 0; i < N; i++){
        double rx_abs_max = MAGMA_D_ZERO;
        double update = MAGMA_D_ZERO;
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
        bool zero_pivot = (rx_abs_max == MAGMA_D_ZERO);
        linfo  = ( zero_pivot && linfo == 0) ? (i+1) : linfo;
        update = ( zero_pivot ) ? MAGMA_D_ZERO : MAGMA_D_ONE;

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = i; j < N; j++){
                sx[j] = update * rA[j];
            }
            sB[0] = rB;
        }
        else if(rowid == i){
            rowid = max_id;
        }
        magmablas_syncwarp(item_ct1);

        /*
        DPCT1064:509: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        reg = (zero_pivot) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i]);
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++) {
                rA[j] -= rA[i] * sx[j];
            }
            rB -= rA[i] * sB[0];
        }
        magmablas_syncwarp(item_ct1);
    }

    sB[rowid] = rB;
    #pragma unroll
    for(int i = N-1; i >= 0; i--) {
        sx[rowid] = rA[i];
        magmablas_syncwarp(item_ct1);
        reg      = MAGMA_Z_DIV(sB[ i ], sx[ i ]);
        sB[ tx ] = (tx <  i) ? sB[ tx ] - reg * sx[ tx ]: sB[ tx ];
        sB[ tx ] = (tx == i) ? reg : sB[ tx ];
        magmablas_syncwarp(item_ct1);
    }
}

template<int N>
void
zgesv_batched_small_kernel(
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array , sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int batchid = item_ct1.get_group(2);

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magma_int_t* ipiv = dipiv_array[batchid];

    /*
    DPCT1064:510: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    int linfo = 0, rowid = tx;

    /*
    DPCT1064:511: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rB = MAGMA_Z_ZERO;
    magmaDoubleComplex *sB = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex *sx = sB + N;
    double* dsx = (double*)(sx + N);
    int* sipiv = (int*)(dsx + N);

    // read
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }
    rB = dB[tx];

    zgesv_batched_small_device<N>(tx, rA, sipiv, rB, sB, sx, dsx, rowid, linfo,
                                  item_ct1);

    magmablas_syncwarp(item_ct1);
    if(tx == 0){
        dinfo_array[batchid] = (magma_int_t)( linfo );
    }

    ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);    // fortran indexing
    dB[ tx ]   = sB[tx];
    #pragma unroll
    for(int i = 0; i < N; i++){
        dA[ i * ldda + rowid ] = rA[i];
    }
}


void
zgesv_batched_small_sm_kernel(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array , sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int batchid = item_ct1.get_group(2);

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magma_int_t* ipiv      = dipiv_array[batchid];
    magma_int_t* info      = &dinfo_array[batchid];

    /*
    DPCT1064:514: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex reg = MAGMA_Z_ZERO;
    /*
    DPCT1064:515: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex update = MAGMA_Z_ZERO;

    int max_id;
    int linfo = 0;
    double rx_abs_max = MAGMA_D_ZERO;

    const int slda = SLDA(n);
    const int sldb = SLDA(n);
    magmaDoubleComplex *sA = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex *sB = sA + slda * n;
    magmaDoubleComplex *sx = sB + sldb * nrhs;
    double* dsx = (double*)(sx + n);
    int* sipiv  = (int*)(dsx + n);

    for(int i = 0; i < n; i++){
        sA(tx,i) = dA[ i * ldda + tx ];
    }

    for(int i = 0; i < nrhs; i++) {
        sB(tx,i) = dB[ i * lddb + tx ];
    }
    /*
    DPCT1065:512: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for(int i = 0; i < n; i++) {
        // izamax and find pivot
        dsx[tx] = sycl::fabs(MAGMA_Z_REAL(sA(tx, i))) + sycl::fabs(MAGMA_Z_IMAG(sA(tx, i)));
        /*
        DPCT1065:516: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        rx_abs_max = dsx[i];
        max_id = i;
        for(int j = i+1; j < n; j++){
            if( dsx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = dsx[j];
            }
        }
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        /*
        DPCT1064:519: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        update = (rx_abs_max == MAGMA_D_ZERO) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;

        // write pivot index
        if(tx == 0){
            sipiv[i] = max_id;
        }

        // swap
        if( max_id != i) {
            reg            = sA(i, tx);
            sA(i, tx)      = sA(max_id, tx);
            sA(max_id, tx) = reg;

            for (int itx = tx; itx < nrhs; itx += item_ct1.get_local_range(2)) {
                reg             = sB(i, itx);
                sB(i, itx)      = sB(max_id, itx);
                sB(max_id, itx) = reg;
            }
        }
        /*
        DPCT1065:517: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        /*
        DPCT1064:520: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        reg = (rx_abs_max == MAGMA_D_ZERO) ? MAGMA_Z_ONE
                                           : MAGMA_Z_DIV(MAGMA_Z_ONE, sA(i, i));
        // scal and ger
        if( tx > i ){
            sA(tx,i) *= reg;
            for(int j = i+1; j < n; j++) {
                sA(tx, j) -= sA(tx, i) * ( update * sA(i, j) );
            }

            for(int j = 0; j < nrhs; j++) {
                sB(tx, j) -= sA(tx, i) * ( update * sB(i, j) );
            }
        }
        /*
        DPCT1065:518: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write A and pivot
    ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);    // fortran indexing
    for(int i = 0; i < n; i++){
        dA[ i * ldda + tx ] = sA(tx, i);
    }

    for(int i = n-1; i >= 0; i--) {
        for(int j = 0; j < nrhs; j++) {
            reg       = MAGMA_Z_DIV(sB(i, j), sA(i,i));
            /*
            DPCT1065:521: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            sB(tx, j) = (tx <  i) ? sB(tx, j) - reg * sA(tx,i): sB(tx, j);
            sB(tx, j) = (tx == i) ? reg : sB(tx, j);
            /*
            DPCT1065:522: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }

    // write
    /*
    DPCT1065:513: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for(int j = 0; j < nrhs; j++) {
        dB[j * lddb + tx] = sB(tx, j);
    }
}

#undef sA
#undef sB

/***************************************************************************//**
    Purpose
    -------
    ZGESV solves a system of linear equations
       A * X = B
    where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as
       A = P * L * U,
    where P is a permutation matrix, L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    This is a batched version that solves batchCount N-by-N matrices in parallel.
    dA, dB, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

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
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in,out]
    dB_array   Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDB,NRHS).
            On entry, each pointer is an right hand side matrix B.
            On exit, each pointer is the solution matrix X.


    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).


    @param[out]
    dinfo_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
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

    @ingroup magma_gesv_batched
*******************************************************************************/
extern "C" magma_int_t magma_zgesv_batched_small(
    magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array,
    magma_int_t ldda, magma_int_t **dipiv_array, magmaDoubleComplex **dB_array,
    magma_int_t lddb, magma_int_t *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue) try {
    magma_int_t arginfo = 0;

    if( n < 0 ) {
        arginfo = -1;
    }
    else if (nrhs < 0) {
        arginfo = -2;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || nrhs == 0) return 0;

    if( n > MAX_N || nrhs > 1 ) {
        arginfo = -100;
        return arginfo;
    }

    const int use_shmem_kernel = (n > 32) ? 1 : 0;
    const magma_int_t slda  = SLDA(n);
    const magma_int_t sldb  = SLDA(n);
    magma_int_t shmem  = 0;
    if(use_shmem_kernel == 1) {
        shmem += (slda * n)    * sizeof(magmaDoubleComplex);    // A
        shmem += (sldb * nrhs) * sizeof(magmaDoubleComplex);    // B
        shmem += n             * sizeof(magmaDoubleComplex);    // sx
        shmem += n             * sizeof(double);                // dsx
        shmem += n             * sizeof(int);                   // pivot
    }
    else {
        shmem += n * sizeof(magmaDoubleComplex); // B
        shmem += n * sizeof(magmaDoubleComplex); // sx
        shmem += n * sizeof(double);             // dsx
        shmem += n * sizeof(int);                // pivot
    }

    const magma_int_t thread_x = n;
    sycl::range<3> threads(1, 1, thread_x);
    sycl::range<3> grid(1, 1, batchCount);

    // TODO: fix error handling for SYCL (exception --> MAGMA error)
    int e = 0;
    if(use_shmem_kernel == 1) {
        magma_device_t device;
        int nthreads_max, shmem_max;
        nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
        shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();
        if ( thread_x > nthreads_max || shmem > shmem_max ) {
            arginfo = -100;
        }
        else {
            void *kernel_args[] = {&n, &nrhs, &dA_array, &ldda, &dipiv_array, &dB_array, &lddb, &dinfo_array};
            /*
            DPCT1049:526: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto n_ct0 = *(magma_int_t *)kernel_args[0];
                        auto nrhs_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dA_array_ct2 =
                            *(magmaDoubleComplex ***)kernel_args[2];
                        auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                        auto dipiv_array_ct4 = *(magma_int_t ***)kernel_args[4];
                        auto dB_array_ct5 =
                            *(magmaDoubleComplex ***)kernel_args[5];
                        auto lddb_ct6 = *(magma_int_t *)kernel_args[6];
                        auto dinfo_array_ct7 = *(magma_int_t **)kernel_args[7];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_sm_kernel(
                                    n_ct0, nrhs_ct1, dA_array_ct2, ldda_ct3,
                                    dipiv_array_ct4, dB_array_ct5, lddb_ct6,
                                    dinfo_array_ct7, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
                            });
                    });
        }
    }
    else {
        void *kernel_args[] = {&dA_array, &ldda, &dipiv_array, &dB_array, &lddb, &dinfo_array};
        switch(n){
            /*
            DPCT1049:527: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 1: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<1>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:528: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 2: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<2>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:529: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 3: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<3>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:530: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 4: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<4>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:531: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 5: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<5>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:532: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 6: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<6>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:533: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 7: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<7>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:534: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 8: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<8>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:535: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 9: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<9>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:536: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 10: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<10>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:537: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 11: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<11>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:538: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 12: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<12>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:539: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 13: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<13>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:540: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 14: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<14>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:541: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 15: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<15>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:542: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 16: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<16>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:543: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 17: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<17>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:544: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 18: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<18>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:545: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 19: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<19>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:546: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 20: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<20>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:547: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 21: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<21>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:548: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 22: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<22>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:549: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 23: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<23>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:550: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 24: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<24>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:551: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 25: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<25>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:552: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 26: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<26>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:553: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 27: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<27>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:554: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 28: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<28>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:555: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 29: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<29>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:556: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 30: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<30>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:557: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 31: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<31>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            /*
            DPCT1049:558: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            case 32: ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                        auto dA_array_ct0 =
                            *(magmaDoubleComplex ***)kernel_args[0];
                        auto ldda_ct1 = *(magma_int_t *)kernel_args[1];
                        auto dipiv_array_ct2 = *(magma_int_t ***)kernel_args[2];
                        auto dB_array_ct3 =
                            *(magmaDoubleComplex ***)kernel_args[3];
                        auto lddb_ct4 = *(magma_int_t *)kernel_args[4];
                        auto dinfo_array_ct5 = *(magma_int_t **)kernel_args[5];

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zgesv_batched_small_kernel<32>(
                                    dA_array_ct0, ldda_ct1, dipiv_array_ct2,
                                    dB_array_ct3, lddb_ct4, dinfo_array_ct5,
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                    });
                break;
            default: e = 1;
        }
    }

    /*
    DPCT1000:524: Error handling if-stmt was detected but could not be
    rewritten.
    */
    // TODO
    if (e != 0) {
        /*
        DPCT1001:523: The statement could not be removed.
        */
        arginfo = -100;
    }
    return arginfo;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#undef SLDA

