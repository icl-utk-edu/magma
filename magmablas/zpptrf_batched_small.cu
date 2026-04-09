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

#ifdef MAGMA_HAVE_HIP
#define PPTRF_MAX_THREADS (256)
#else
#define PPTRF_MAX_THREADS (64)
#endif

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
template<int N>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(PPTRF_MAX_THREADS)
#endif
void
zpptrf_lower_batched_small_kernel(
        magmaDoubleComplex** dA_array,
        magma_int_t *info_array, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];
    constexpr int sizeA    = (N+1)*N/2;
    constexpr int sizeA_N  = (sizeA / N) * N;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dA_array[batchid];
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
    // sizeA is always divisible by N, so since we are assuming N threads,
    // we don't need a cleanup section for the loop below
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N){
        dA[ i + tx ] = sA[ i + tx ];
    }

    if(tx < sizeA - sizeA_N) {
        dA[sizeA_N + tx] = sA[sizeA_N + tx];
    }

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
magma_zpptrf_batched_small(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(uplo != MagmaLower)
        arginfo = -1;
    else if( (n < 0) || ( n > 32 ) )
        arginfo = -2;
    else if ( batchCount < 0 )
        arginfo = -5;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || batchCount == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    const magma_int_t ntcol = PPTRF_MAX_THREADS / 64;;
    #else
    const magma_int_t ntcol = PPTRF_MAX_THREADS / 32;
    #endif
    magma_int_t shmem  = ntcol * n * (n+1) * sizeof(magmaDoubleComplex);
    dim3 threads(n, ntcol, 1);
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);

    switch(n){
        #if 0
        case  8: zpptrf_lower_batched_small_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        #else
        case  1: zpptrf_lower_batched_small_kernel< 1><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  2: zpptrf_lower_batched_small_kernel< 2><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  3: zpptrf_lower_batched_small_kernel< 3><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  4: zpptrf_lower_batched_small_kernel< 4><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  5: zpptrf_lower_batched_small_kernel< 5><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  6: zpptrf_lower_batched_small_kernel< 6><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  7: zpptrf_lower_batched_small_kernel< 7><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  8: zpptrf_lower_batched_small_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case  9: zpptrf_lower_batched_small_kernel< 9><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 10: zpptrf_lower_batched_small_kernel<10><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 11: zpptrf_lower_batched_small_kernel<11><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 12: zpptrf_lower_batched_small_kernel<12><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 13: zpptrf_lower_batched_small_kernel<13><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 14: zpptrf_lower_batched_small_kernel<14><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 15: zpptrf_lower_batched_small_kernel<15><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 16: zpptrf_lower_batched_small_kernel<16><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 17: zpptrf_lower_batched_small_kernel<17><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 18: zpptrf_lower_batched_small_kernel<18><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 19: zpptrf_lower_batched_small_kernel<19><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 20: zpptrf_lower_batched_small_kernel<20><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 21: zpptrf_lower_batched_small_kernel<21><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 22: zpptrf_lower_batched_small_kernel<22><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 23: zpptrf_lower_batched_small_kernel<23><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 24: zpptrf_lower_batched_small_kernel<24><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 25: zpptrf_lower_batched_small_kernel<25><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 26: zpptrf_lower_batched_small_kernel<26><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 27: zpptrf_lower_batched_small_kernel<27><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 28: zpptrf_lower_batched_small_kernel<28><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 29: zpptrf_lower_batched_small_kernel<29><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 30: zpptrf_lower_batched_small_kernel<30><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 31: zpptrf_lower_batched_small_kernel<31><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        case 32: zpptrf_lower_batched_small_kernel<32><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, info_array, batchCount); break;
        #endif
        default: fprintf(stderr, "MAGMA: error in *pptrf_batched_small, unsupported size '%lld'\n", (long long)n);
    }
    return arginfo;
}
