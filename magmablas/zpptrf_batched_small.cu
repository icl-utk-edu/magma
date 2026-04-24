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
template<int N, int MAX_THREADS>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(MAX_THREADS)
#endif
void
zpptrf_lower_batched_small_kernel(
        magmaDoubleComplex** dAP_array,
        magma_int_t *info_array, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];
    constexpr int sizeA    = (N+1)*N/2;
    constexpr int sizeA_N  = (sizeA / N) * N;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dAP_array[batchid];
    magma_int_t* info = &info_array[batchid];

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;

    int linfo = 0;
    sA += ty * sizeA;

    // read
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N){
        sA[ i + tx ] = dA[ i + tx ];
    }

    if(tx < sizeA - sizeA_N) {
        sA[sizeA_N + tx] = dA[sizeA_N + tx];
    }
    magmablas_syncwarp();

    #pragma unroll
    for(int j = 0; j < N; j++){
        // linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        // update column j
        if(tx >= j) {
            #pragma unroll
            for(int k = 0; k < j; k++) {
                sA(tx, j) -= sA(tx, k) * MAGMA_Z_CONJ( sA(j,k) );
            }
        }
        magmablas_syncwarp();

        // factorize column j
        if(tx == j) {
            sA(j, j) = MAGMA_Z_MAKE(sqrt(MAGMA_Z_REAL(sA(j, j))), MAGMA_D_ZERO);
        }
        magmablas_syncwarp();

        if(tx > j) {
            sA(tx, j) *= MAGMA_Z_DIV( MAGMA_Z_ONE, sA(j, j) );
        }
        magmablas_syncwarp();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N){
        dA[ i + tx ] = sA[ i + tx ];
    }

    if(tx < sizeA - sizeA_N) {
        dA[sizeA_N + tx] = sA[sizeA_N + tx];
    }

}

////////////////////////////////////////////////////////////////////////////////
__global__ void
zpptrf_lower_batched_small_kernel_n(
        magma_int_t n,
        magmaDoubleComplex** dAP_array,
        magma_int_t *info_array, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int sizeA    = (n+1)*n/2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int  N = n; // for the sA macro defined above
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dAP_array[batchid];
    magma_int_t* info = &info_array[batchid];

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;

    int linfo = 0;
    sA += ty * sizeA;

    // read
    for(int i = tx; i < sizeA; i+=n){
        sA[ i ] = dA[ i ];
    }
    __syncthreads();

    for(int j = 0; j < n; j++){
        // update column j
        if(tx >= j) {
            for(int k = 0; k < j; k++) {
                sA(tx, j) -= sA(tx, k) * MAGMA_Z_CONJ( sA(j,k) );
            }
        }
        __syncthreads();

        // factorize column j
        if(tx == j) {
            sA(j, j) = MAGMA_Z_MAKE(sqrt(MAGMA_Z_REAL(sA(j, j))), MAGMA_D_ZERO);
        }
        __syncthreads();

        if(tx > j) {
            sA(tx, j) *= MAGMA_Z_DIV( MAGMA_Z_ONE, sA(j, j) );
        }
        __syncthreads();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write
    for(int i = tx; i < sizeA; i+=n){
        dA[ i ] = sA[ i ];
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int N, int MAX_THREADS>
magma_int_t
zpptrf_lower_batched_small_kernel_driver(
    magma_uplo_t uplo,
    magmaDoubleComplex** dAP_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if(uplo != MagmaLower)
        arginfo = -1;
    else if( (N < 0) || ( N > 64 ) )
        arginfo = -2;
    else if ( batchCount < 0 )
        arginfo = -5;

    if (arginfo != 0) {
        return arginfo;
    }

    if( N == 0 || batchCount == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    magma_int_t ntcol = min(1, MAX_THREADS / 64);
    #else
    magma_int_t ntcol = min(1, MAX_THREADS / 32);
    #endif

    magma_int_t shmem = ntcol * (N * (N+1) / 2) * sizeof(magmaDoubleComplex);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 threads(N, ntcol, 1);
    dim3 grid(gridx, 1, 1);

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zpptrf_lower_batched_small_kernel<N, MAX_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }

    void *kernel_args[] = {&dAP_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zpptrf_lower_batched_small_kernel<N, MAX_THREADS>, grid, threads, kernel_args, shmem, queue->cuda_stream());

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
magma_zpptrf_batched_small(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dAP_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(uplo != MagmaLower)
        arginfo = -1;
    else if( (n < 0) || ( n > 64 ) )
        arginfo = -2;
    else if ( batchCount < 0 )
        arginfo = -5;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || batchCount == 0 ) return 0;

    switch(n){
        case  1: arginfo = zpptrf_lower_batched_small_kernel_driver< 1, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  2: arginfo = zpptrf_lower_batched_small_kernel_driver< 2, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  3: arginfo = zpptrf_lower_batched_small_kernel_driver< 3, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  4: arginfo = zpptrf_lower_batched_small_kernel_driver< 4, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  5: arginfo = zpptrf_lower_batched_small_kernel_driver< 5, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  6: arginfo = zpptrf_lower_batched_small_kernel_driver< 6, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  7: arginfo = zpptrf_lower_batched_small_kernel_driver< 7, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  8: arginfo = zpptrf_lower_batched_small_kernel_driver< 8, 128>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case  9: arginfo = zpptrf_lower_batched_small_kernel_driver< 9,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 10: arginfo = zpptrf_lower_batched_small_kernel_driver<10,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 11: arginfo = zpptrf_lower_batched_small_kernel_driver<11,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 12: arginfo = zpptrf_lower_batched_small_kernel_driver<12,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 13: arginfo = zpptrf_lower_batched_small_kernel_driver<13,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 14: arginfo = zpptrf_lower_batched_small_kernel_driver<14,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 15: arginfo = zpptrf_lower_batched_small_kernel_driver<15,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 16: arginfo = zpptrf_lower_batched_small_kernel_driver<16,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 17: arginfo = zpptrf_lower_batched_small_kernel_driver<17,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 18: arginfo = zpptrf_lower_batched_small_kernel_driver<18,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 19: arginfo = zpptrf_lower_batched_small_kernel_driver<19,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 20: arginfo = zpptrf_lower_batched_small_kernel_driver<20,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 21: arginfo = zpptrf_lower_batched_small_kernel_driver<21,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 22: arginfo = zpptrf_lower_batched_small_kernel_driver<22,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 23: arginfo = zpptrf_lower_batched_small_kernel_driver<23,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 24: arginfo = zpptrf_lower_batched_small_kernel_driver<24,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 25: arginfo = zpptrf_lower_batched_small_kernel_driver<25,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 26: arginfo = zpptrf_lower_batched_small_kernel_driver<26,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 27: arginfo = zpptrf_lower_batched_small_kernel_driver<27,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 28: arginfo = zpptrf_lower_batched_small_kernel_driver<28,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 29: arginfo = zpptrf_lower_batched_small_kernel_driver<29,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 30: arginfo = zpptrf_lower_batched_small_kernel_driver<30,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 31: arginfo = zpptrf_lower_batched_small_kernel_driver<31,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        case 32: arginfo = zpptrf_lower_batched_small_kernel_driver<32,  64>(uplo, dAP_array, info_array, batchCount, queue ); break;
        default: arginfo = -100;
    }

    if(arginfo != 0) {
        arginfo = 0;
        magma_int_t ntcol = 1;
        magma_int_t shmem = ntcol * (n * (n+1) / 2) * sizeof(magmaDoubleComplex);
        magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
        dim3 threads(n, ntcol, 1);
        dim3 grid(gridx, 1, 1);

        void *kernel_args[] = {&n, &dAP_array, &info_array, &batchCount};
        cudaError_t e = cudaLaunchKernel((void*)zpptrf_lower_batched_small_kernel_n, grid, threads, kernel_args, shmem, queue->cuda_stream());

        if( e != cudaSuccess ) {
            //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
            arginfo = -100;
        }
    }

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    PPTF2 computes the Cholesky factorization in blocked algorithm. 

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
magma_zpptf2_batched_small(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dAP_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(uplo != MagmaLower)
        arginfo = -1;
    else if( (n < 0) || ( n > 64 ) )
        arginfo = -2;
    else if ( batchCount < 0 )
        arginfo = -5;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || batchCount == 0 ) return 0;

    arginfo = magma_zpptrf_lpout_batched(uplo, n, dAP_array, 0, 0, n, 0, info_array, batchCount, queue);
  
    return arginfo;

}
