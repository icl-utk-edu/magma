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

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.cuh"
#include "shuffle.cuh"
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
__device__ __inline__ void
zgesv_batched_small_device(
    const int tx,
    magmaDoubleComplex rA[N], int* sipiv,
    magmaDoubleComplex &rB, magmaDoubleComplex *sB,
    magmaDoubleComplex *sx, double *dsx,
    int &rowid, int &linfo )
{
    magmaDoubleComplex reg    = MAGMA_Z_ZERO;
    int max_id;

    #pragma unroll
    for(int i = 0; i < N; i++){
        double rx_abs_max = MAGMA_D_ZERO;
        double update = MAGMA_D_ZERO;
        // izamax and find pivot
        dsx[ rowid ] = fabs(MAGMA_Z_REAL( rA[i] )) + fabs(MAGMA_Z_IMAG( rA[i] ));
        magmablas_syncwarp();
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
        magmablas_syncwarp();

        reg = ( zero_pivot ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++) {
                rA[j] -= rA[i] * sx[j];
            }
            rB -= rA[i] * sB[0];
        }
        magmablas_syncwarp();
    }

    sB[rowid] = rB;
    #pragma unroll
    for(int i = N-1; i >= 0; i--) {
        sx[rowid] = rA[i];
        magmablas_syncwarp();
        reg      = MAGMA_Z_DIV(sB[ i ], sx[ i ]);
        sB[ tx ] = (tx <  i) ? sB[ tx ] - reg * sx[ tx ]: sB[ tx ];
        sB[ tx ] = (tx == i) ? reg : sB[ tx ];
        magmablas_syncwarp();
    }
}

template<int N>
__global__ void
zgesv_batched_small_kernel(
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array )
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int tx = threadIdx.x;
    const int batchid = blockIdx.x ;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magma_int_t* ipiv = dipiv_array[batchid];

    magmaDoubleComplex rA[N]  = {MAGMA_Z_ZERO};
    int linfo = 0, rowid = tx;

    magmaDoubleComplex  rB = MAGMA_Z_ZERO;
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

    zgesv_batched_small_device<N>( tx, rA, sipiv, rB, sB, sx, dsx, rowid, linfo );

    magmablas_syncwarp();
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


__global__ void
zgesv_batched_small_sm_kernel(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array )
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int tx = threadIdx.x;
    const int batchid = blockIdx.x ;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magma_int_t* ipiv      = dipiv_array[batchid];
    magma_int_t* info      = &dinfo_array[batchid];

    magmaDoubleComplex reg    = MAGMA_Z_ZERO;
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
    __syncthreads();

    #pragma unroll
    for(int i = 0; i < n; i++) {
        // izamax and find pivot
        dsx[ tx ] = fabs(MAGMA_Z_REAL( sA(tx,i) )) + fabs(MAGMA_Z_IMAG( sA(tx,i) ));
        __syncthreads();
        rx_abs_max = dsx[i];
        max_id = i;
        for(int j = i+1; j < n; j++){
            if( dsx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = dsx[j];
            }
        }
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        update = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;

        // write pivot index
        if(tx == 0){
            sipiv[i] = max_id;
        }

        // swap
        if( max_id != i) {
            reg            = sA(i, tx);
            sA(i, tx)      = sA(max_id, tx);
            sA(max_id, tx) = reg;

            for(int itx = tx; itx < nrhs; itx+=blockDim.x) {
                reg             = sB(i, itx);
                sB(i, itx)      = sB(max_id, itx);
                sB(max_id, itx) = reg;
            }
        }
        __syncthreads();

        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sA(i,i) );
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
        __syncthreads();
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
            __syncthreads();
            sB(tx, j) = (tx <  i) ? sB(tx, j) - reg * sA(tx,i): sB(tx, j);
            sB(tx, j) = (tx == i) ? reg : sB(tx, j);
            __syncthreads();
        }
    }

    // write
    __syncthreads();
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
extern "C" magma_int_t
magma_zgesv_batched_small(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array,
    magma_int_t batchCount, magma_queue_t queue )
{
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
    dim3 threads(thread_x, 1, 1);
    dim3 grid(batchCount, 1, 1);

    cudaError_t e = cudaErrorInvalidValue;
    if(use_shmem_kernel == 1) {
        magma_device_t device;
        int nthreads_max, shmem_max;
        magma_getdevice( &device );
        cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
        #if CUDA_VERSION >= 9000
        cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (shmem <= shmem_max) {
            cudaFuncSetAttribute(zgesv_batched_small_sm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        }
        #else
        cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
        #endif    // CUDA_VERSION >= 9000

        if ( thread_x > nthreads_max || shmem > shmem_max ) {
            arginfo = -100;
        }
        else {
            void *kernel_args[] = {&n, &nrhs, &dA_array, &ldda, &dipiv_array, &dB_array, &lddb, &dinfo_array};
            e = cudaLaunchKernel((void*)zgesv_batched_small_sm_kernel, grid, threads, kernel_args, shmem, queue->cuda_stream());
        }
    }
    else {
        void *kernel_args[] = {&dA_array, &ldda, &dipiv_array, &dB_array, &lddb, &dinfo_array};
        switch(n){
            case  1: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 1>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  2: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 2>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  3: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 3>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  4: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 4>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  5: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 5>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  6: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 6>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  7: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 7>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  8: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 8>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case  9: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel< 9>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 10: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<10>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 11: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<11>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 12: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<12>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 13: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<13>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 14: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<14>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 15: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<15>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 16: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<16>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 17: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<17>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 18: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<18>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 19: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<19>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 20: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<20>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 21: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<21>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 22: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<22>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 23: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<23>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 24: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<24>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 25: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<25>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 26: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<26>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 27: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<27>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 28: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<28>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 29: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<29>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 30: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<30>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 31: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<31>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            case 32: e = cudaLaunchKernel((void*)zgesv_batched_small_kernel<32>, grid, threads, kernel_args, shmem, queue->cuda_stream()); break;
            default: e = cudaErrorInvalidValue;
        }
    }

    if( e != cudaSuccess ) {
        arginfo = -100;
    }
    return arginfo;
}

#undef SLDA

