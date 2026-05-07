/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.cuh"
#include "batched_kernel_param.h"

#define PRECISION_z
#define DBG

// formula for lower part access
#define sA(i, j) sA[N*j - j*(j+1)/2 + i]

// leading dimension in shared memory
#define SLDB(n)    ( ((n+1)%4) == 0 ? (n) : (n+1) )

// regular access for dB/sB
#define dB(i, j) dB[(j) * lddb + (i)]
#define sB(i, j) sB[(j) * sldb + (i)]

#define ZPPTRS_KERNEL_MAX_THREADS (64)

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ void print_memory(
                const char* msg,
                int m, int n, T* sA, int lda,
                int tx, int ty, int tz,
                int bx, int by, int bz)
{
#if defined(PRECISION_d) && defined(DBG)
    __syncthreads();
    if(threadIdx.x == tx && threadIdx.y == ty && threadIdx.z == tz &&
       blockIdx.x  == bx && blockIdx.y  == by && blockIdx.z  == bz) {
        printf("%s = [ \n", msg);
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                printf("%8.4f  ", (double)(sA[j*lda+i]));
            }
            printf("\n");
        }
        printf("]; \n");
    }
    __syncthreads();
#endif
}

////////////////////////////////////////////////////////////////////////////////
// N is the size of the packed matrix
// NTHREADS >= N
template<int N, int NTHREADS>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(NTHREADS)
#endif
void
zpptrs_lower_batched_small_kernel(
        int nrhs,
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex** dB_array, int lddb,
        int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    // NRHS_NB is a blocking size for right hand sides (NRHS >= N)
    constexpr int NRHS_NB = NTHREADS;

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;
    constexpr int sizeA_NRHS      = ( sizeA / NRHS_NB ) * NRHS_NB;

    constexpr int sldb          = SLDB(N);

    const int tx  = threadIdx.x;
    const int bx  = blockIdx.x;
    const int batchid = blockIdx.z;

    magmaDoubleComplex* dA = dAP_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    // advance dB
    dB += bx * NRHS_NB * lddb;

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB = sA + sizeA_aligned;

    magmaDoubleComplex rB[N] = {MAGMA_Z_ZERO};

    const int local_nrhs = min(NRHS_NB, nrhs - bx * NRHS_NB);

    // read A
    #pragma unroll
    for(int i = 0; i < sizeA_NRHS; i+=NRHS_NB){
        sA[ i + tx ] = dA[ i + tx ];
    }

    if(tx < (sizeA - sizeA_NRHS)) {
        sA[sizeA_NRHS + tx] = dA[sizeA_NRHS + tx];
    }

    // read B
    if(tx < N) {
        for(int i = 0; i < local_nrhs; i++) {
            sB(tx, i) = dB(tx, i);
        }
    }
    __syncthreads();

    // let each thread read one column of B
    // which enables the solve to completely independent of other threads
    if(tx < local_nrhs) {
        #pragma unroll
        for(int i = 0; i < N; i++) {
            rB[i] = sB(i, tx);
        }
    }
    __syncthreads();

    //print_memory( "sA", sizeA, 1, sA, sizeA, 0, 0, 0, 0, 0, 0);
    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    // Solving L L^T x = b
    // First, solve L y = b for y
    #pragma unroll
    for(int i = 0; i < N; i++) {
        magmaDoubleComplex rTmp = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(i,i));
        rB[i] *= rTmp;
        #pragma unroll
        for(int j = i+1; j < N; j++) {
            rB[j] -= rB[i] * sA(j,i);
        }
    }

    // Second, solve L^T x = y for x
    #pragma unroll
    for(int i = N-1; i >= 0; i--) {
        magmaDoubleComplex rTmp = MAGMA_Z_DIV(MAGMA_Z_ONE, MAGMA_Z_CONJ(sA(i,i)) );
        rB[i] *= rTmp;
        #pragma unroll
        for(int j = i-1; j >= 0; j--) {
            rB[j] -= rB[i] * MAGMA_Z_CONJ(sA(i,j));
        }
    }

    // write B
    if(tx < local_nrhs) {
        #pragma unroll
        for(int i = 0; i < N; i++) {
            sB(i, tx) = rB[i];
        }
    }
    __syncthreads();

    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    if(tx < N) {
        for(int i = 0; i < local_nrhs; i++) {
            dB(tx, i) = sB(tx, i);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(ZPPTRS_KERNEL_MAX_THREADS)
#endif
void
zpptrs_lower_batched_small_kernel_n(
        int n, int nrhs, int nrhs_nb,
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex** dB_array, int lddb,
        int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    const int N               = n; // for the sA macro expansion
    const int alignment_bytes = 128;
    const int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    const int sizeA           = (n+1)*n/2;
    const int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;
    const int sizeA_nrhs      = ( sizeA / nrhs_nb ) * nrhs_nb;

    const int sldb          = SLDB(n);

    const int tx  = threadIdx.x;
    const int bx  = blockIdx.x;
    const int batchid = blockIdx.z;

    magmaDoubleComplex* dA = dAP_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    // advance dB
    dB += bx * nrhs_nb * lddb;

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB = sA + sizeA_aligned;

    const int local_nrhs = min(nrhs_nb, nrhs - bx * nrhs_nb);

    // read A
    for(int i = 0; i < sizeA_nrhs; i+=nrhs_nb){
        sA[ i + tx ] = dA[ i + tx ];
    }

    if(tx < (sizeA - sizeA_nrhs)) {
        sA[sizeA_nrhs + tx] = dA[sizeA_nrhs + tx];
    }

    // read B
    if(tx < n) {
        for(int i = 0; i < local_nrhs; i++) {
            sB(tx, i) = dB(tx, i);
        }
    }
    __syncthreads();

    //print_memory( "sA", sizeA, 1, sA, sizeA, 0, 0, 0, 0, 0, 0);
    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    // Solving L L^T x = b
    // First, solve L y = b for y (in shared memory)
    // each thread handles one column, so no need to sync
    for(int i = 0; i < n; i++) {
        magmaDoubleComplex rTmp = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(i,i));
        sB(i,tx) *= rTmp;
        for(int j = i+1; j < N; j++) {
            sB(j,tx) -= sB(i,tx) * sA(j,i);
        }
    }

    // Second, solve L^T x = y for x (in shared memory)
    // each thread handles one column, so no need to sync
    for(int i = n-1; i >= 0; i--) {
        magmaDoubleComplex rTmp = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(i,i));
        sB(i, tx) *= rTmp;
        for(int j = i-1; j >= 0; j--) {
            sB(j,tx) -= sB(i, tx) * sA(i,j);
        }
    }
    __syncthreads();

    // write B
    if(tx < n) {
        for(int i = 0; i < local_nrhs; i++) {
            dB(tx, i) = sB(tx, i);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int N, int NTHREADS>
magma_int_t
zpptrs_lower_batched_small_kernel_driver(
    int nrhs,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    int batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if(nrhs < 0)
        arginfo = -1;
    else if( lddb < N )
        arginfo = -4;
    else if ( batchCount < 0 )
        arginfo = -5;

    if (arginfo != 0) {
        return arginfo;
    }

    if( N == 0 || nrhs == 0 || batchCount == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    magma_int_t ntcol = 1; //min(1, NRHS_NB / 64);
    #else
    magma_int_t ntcol = 1; //min(1, NRHS_NB / 32);
    #endif

    // constants
    constexpr int NRHS_NB         = NTHREADS;  // NRHS_NB is a blocking size for right hand sides (NRHS >= N)
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;


    constexpr int sldb          = SLDB(N);
    constexpr int sizeB         = sldb * NRHS_NB;
    constexpr int sizeB_aligned = ( (sizeB + alignment - 1) / alignment) * alignment;

    // configure shared memory
    magma_int_t shmem = 0;
    shmem += ntcol * sizeA_aligned * sizeof(magmaDoubleComplex);
    shmem += ntcol * sizeB_aligned * sizeof(magmaDoubleComplex);

    magma_int_t gridx = magma_ceildiv(nrhs, NRHS_NB);
    dim3 threads(NRHS_NB, 1, 1);
    dim3 grid(gridx, 1, batchCount);

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zpptrs_lower_batched_small_kernel<N, NRHS_NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }

    void *kernel_args[] = {&nrhs, &dAP_array, &dB_array, &lddb, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zpptrs_lower_batched_small_kernel<N, NRHS_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());

    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }


    return arginfo;
}
/***************************************************************************//**
    Purpose
    -------
    PPTRF computes

    This is a batched version that factors batchCount N-by-N matrices in parallel.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dAP_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i,

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpptrs_batched_small(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(n < 0 || n > 64)
        arginfo = -1;
    else if( nrhs < 0 )
        arginfo = -2;
    else if( lddb < n )
        arginfo = -5;
    else if ( batchCount < 0 )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || nrhs == 0 || batchCount == 0 ) return 0;

    switch(n){
        case  1: arginfo = zpptrs_lower_batched_small_kernel_driver< 1, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  2: arginfo = zpptrs_lower_batched_small_kernel_driver< 2, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  3: arginfo = zpptrs_lower_batched_small_kernel_driver< 3, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  4: arginfo = zpptrs_lower_batched_small_kernel_driver< 4, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  5: arginfo = zpptrs_lower_batched_small_kernel_driver< 5, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  6: arginfo = zpptrs_lower_batched_small_kernel_driver< 6, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  7: arginfo = zpptrs_lower_batched_small_kernel_driver< 7, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  8: arginfo = zpptrs_lower_batched_small_kernel_driver< 8, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  9: arginfo = zpptrs_lower_batched_small_kernel_driver< 9, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 10: arginfo = zpptrs_lower_batched_small_kernel_driver<10, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 11: arginfo = zpptrs_lower_batched_small_kernel_driver<11, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 12: arginfo = zpptrs_lower_batched_small_kernel_driver<12, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 13: arginfo = zpptrs_lower_batched_small_kernel_driver<13, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 14: arginfo = zpptrs_lower_batched_small_kernel_driver<14, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 15: arginfo = zpptrs_lower_batched_small_kernel_driver<15, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 16: arginfo = zpptrs_lower_batched_small_kernel_driver<16, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 17: arginfo = zpptrs_lower_batched_small_kernel_driver<17, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 18: arginfo = zpptrs_lower_batched_small_kernel_driver<18, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 19: arginfo = zpptrs_lower_batched_small_kernel_driver<19, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 20: arginfo = zpptrs_lower_batched_small_kernel_driver<20, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 21: arginfo = zpptrs_lower_batched_small_kernel_driver<21, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 22: arginfo = zpptrs_lower_batched_small_kernel_driver<22, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 23: arginfo = zpptrs_lower_batched_small_kernel_driver<23, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 24: arginfo = zpptrs_lower_batched_small_kernel_driver<24, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 25: arginfo = zpptrs_lower_batched_small_kernel_driver<25, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 26: arginfo = zpptrs_lower_batched_small_kernel_driver<26, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 27: arginfo = zpptrs_lower_batched_small_kernel_driver<27, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 28: arginfo = zpptrs_lower_batched_small_kernel_driver<28, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 29: arginfo = zpptrs_lower_batched_small_kernel_driver<29, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 30: arginfo = zpptrs_lower_batched_small_kernel_driver<30, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 31: arginfo = zpptrs_lower_batched_small_kernel_driver<31, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 32: arginfo = zpptrs_lower_batched_small_kernel_driver<32, 32>(nrhs, dAP_array, dB_array, lddb, batchCount, queue ); break;
        default: arginfo = -100;
    }

    if(arginfo != 0) {
        arginfo = 0;

        magma_device_t device;
        magma_getdevice( &device );

        // constants
        const int alignment_bytes = 128;
        const int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
        const int sizeA           = (n+1)*n/2;
        const int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;

        int nrhs_nb = n;  // nrhs_nb is a blocking size for right hand sides (NRHS >= N)

        const int sldb          = SLDB(n);
        const int sizeB         = sldb * nrhs_nb;
        const int sizeB_aligned = ( (sizeB + alignment - 1) / alignment) * alignment;

        // configure shared memory
        magma_int_t shmem = 0;
        shmem += sizeA_aligned * sizeof(magmaDoubleComplex);
        shmem += sizeB_aligned * sizeof(magmaDoubleComplex);

        int shmem_max = 0;
        #if CUDA_VERSION >= 9000
        cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (shmem <= shmem_max) {
            cudaFuncSetAttribute(zpptrs_lower_batched_small_kernel_n, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        }
        #else
        cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
        #endif    // CUDA_VERSION >= 9000

        if ( shmem > shmem_max ) {
            arginfo = -200;;
        }
        else {
            // configure grid and threads
            magma_int_t gridx = magma_ceildiv(nrhs, nrhs_nb);
            dim3 threads(nrhs_nb, 1, 1);
            dim3 grid(gridx, 1, batchCount);

            void *kernel_args[] = {&n, &nrhs, &nrhs_nb, &dAP_array, &dB_array, &lddb, &batchCount};
            cudaError_t e = cudaLaunchKernel((void*)zpptrs_lower_batched_small_kernel_n, grid, threads, kernel_args, shmem, queue->cuda_stream());

            if( e != cudaSuccess ) {
                //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
                arginfo = -300;
            }
        }
    }

    return arginfo;
}
