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
#include "tune_ppinv_apply.h"
#include <assert.h>

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
// Computes B = A * B
// A is NxN (the inverse of a matrix in packed format), B is Nx1
template<int N, int NTCOL>
__global__
__launch_bounds__(N*NTCOL)
void
zhemv_lower_packed_inplace_batched_small_kernel(
        magmaDoubleComplex** dAPinv_array,
        magmaDoubleComplex** dB_array, int lddb, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_N         = ( sizeA / N ) * N;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;

    int tx       = threadIdx.x;
    int ty       = threadIdx.y;
    int nty      = blockDim.y;
    int batchid  = blockIdx.x * nty + ty;

    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dAPinv_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB = sA + (nty * sizeA_aligned);

    sA += ty * sizeA_aligned;
    sB += ty * N;

    // reg. variables
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    magmaDoubleComplex rC = MAGMA_Z_ZERO;


    // read A (packed in glmem)
    // for now use the first N threads to read
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N)
        sA[ i + tx ] = dA[ i + tx ];

    if(tx < sizeA-sizeA_N) sA[sizeA_N + tx] = dA[sizeA_N + tx];

    sB[ tx ] = dB[ tx ];

    __syncthreads();

    // read sA -> rA (make it Hermitian)
    #pragma unroll
    for(int i = 0; i < N; i++) {
        rA[ i ] = (tx < i) ? MAGMA_Z_CONJ( sA(i,tx) ) : sA(tx,i);
    }

    // multiply
    #pragma unroll
    for(int i = 0; i < N; i++) {
        rC += rA[ i ] * sB[ i ];
    }

    // update B
    dB[ tx ] = rC;
}

////////////////////////////////////////////////////////////////////////////////
template<int N>
magma_int_t
zhemv_packed_inplace_batched_small_kernel_driver(
    magmaDoubleComplex** dAPinv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    int batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if( lddb < N )
        arginfo = -3;
    else if ( batchCount < 0 )
        arginfo = -4;

    if (arginfo != 0) {
        return arginfo;
    }

    if( N == 0 || batchCount == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    constexpr int NTCOL = max(1, 64 / N);
    #else
    constexpr int NTCOL = max(1, 32 / N);
    #endif

    // constants
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;
    constexpr int sizeB           = N;

    // configure shared memory
    magma_int_t shmem = 0;
    shmem += NTCOL * sizeA_aligned * sizeof(magmaDoubleComplex);
    shmem += NTCOL * sizeB         * sizeof(magmaDoubleComplex);

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zhemv_lower_packed_inplace_batched_small_kernel<N, NTCOL>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }
    else {
        dim3 threads(N, NTCOL, 1);
        dim3 grid(magma_ceildiv(batchCount,NTCOL), 1, 1);
        void *kernel_args[] = {&dAPinv_array, &dB_array, &lddb, &batchCount};
        cudaError_t e = cudaLaunchKernel((void*)zhemv_lower_packed_inplace_batched_small_kernel<N, NTCOL>, grid, threads, kernel_args, shmem, queue->cuda_stream());
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
    PPINV_APPLY multilpes the inverse of a Hermitian matrix 'A' by a dense matrix B
    The matrix B is overwritten with the product

    This is a batched version that factors batchCount N-by-N matrices in parallel.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dAPinv_array    Array of pointers, dimension (batchCount).
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
extern "C" void
magma_zhemv_packed_inplace_batched_small(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dAPinv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if( uplo != MagmaLower )
        arginfo = -1;
    else if(n < 0 || n > 32)
        arginfo = -2;
    else if( lddb < max(1,n) )
        arginfo = -5;
    else if ( batchCount < 0 )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    if( n == 0 || batchCount == 0 ) return;

    switch(n){
        case  1: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 1>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  2: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 2>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  3: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 3>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  4: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 4>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  5: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 5>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  6: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 6>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  7: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 7>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  8: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 8>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  9: arginfo = zhemv_packed_inplace_batched_small_kernel_driver< 9>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 10: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<10>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 11: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<11>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 12: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<12>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 13: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<13>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 14: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<14>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 15: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<15>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 16: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<16>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 17: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<17>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 18: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<18>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 19: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<19>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 20: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<20>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 21: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<21>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 22: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<22>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 23: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<23>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 24: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<24>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 25: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<25>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 26: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<26>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 27: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<27>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 28: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<28>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 29: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<29>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 30: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<30>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 31: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<31>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 32: arginfo = zhemv_packed_inplace_batched_small_kernel_driver<32>(dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        default:;
    }
}
