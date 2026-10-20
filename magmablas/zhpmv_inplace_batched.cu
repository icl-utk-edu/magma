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
// Computes x = A * x
// A is NxN symmetric/Hermitian in packed format, x is Nx1
template<int N, int NTCOL>
__global__
__launch_bounds__(N*NTCOL)
void
zhpmv_lower_inplace_batched_small_kernel(
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex** dX_array, int incx, int batchCount)
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

    magmaDoubleComplex* dA = dAP_array[batchid];
    magmaDoubleComplex* dX = dX_array[batchid];

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

    sB[ tx ] = dX[ tx * incx ];

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
    dX[ tx * incx ] = rC;
}

////////////////////////////////////////////////////////////////////////////////
// kernel driver
template<int N>
magma_int_t
zhpmv_inplace_batched_small_kernel_driver(
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dX_array, magma_int_t incx,
    int batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if( incx < 1 )
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
        cudaFuncSetAttribute(zhpmv_lower_inplace_batched_small_kernel<N, NTCOL>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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
        void *kernel_args[] = {&dAP_array, &dX_array, &incx, &batchCount};
        cudaError_t e = cudaLaunchKernel((void*)zhpmv_lower_inplace_batched_small_kernel<N, NTCOL>, grid, threads, kernel_args, shmem, queue->cuda_stream());
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
    ZHPMV overwrites an nx1 vector 'x' by the product A * x, where 'A' is a
    Hermitian matrix of size n, stored in packed format.

    This is a batched version, computing many independent matrix-vector products
    in  parallel.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t.
            On entry, UPLO specifies whether the upper or lower
            triangular part of the array A is stored in the packed
            format:
      -     = MagmaUpper:  Not supported
      -     = MagmaLower:  Only the lower triangular part of A is stored.

    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dAP_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU of size (N+1)*N/2.
            On entry, the upper or lower triangle part of each matrix AP,
            packed columnwise in a linear array.

    @param[in]
    dX_array    Array of pointers, dimension(batchCount).
            Each is a COMPLEX_16 array X of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector X.
            On exit, X is overwritten with the product A*x

    @param[in]
    incx    INTEGER.
            On entry, INCX specifies the increment for the elements of
            X. INCX must not be zero.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" void
magma_zhpmv_inplace_batched_small(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dX_array, magma_int_t incx,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if( uplo != MagmaLower )
        arginfo = -1;
    else if(n < 0 || n > 32)
        arginfo = -2;
    else if( incx < 1 )
        arginfo = -5;
    else if ( batchCount < 0 )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    if( n == 0 || batchCount == 0 ) return;

    switch(n){
        case  1: arginfo = zhpmv_inplace_batched_small_kernel_driver< 1>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  2: arginfo = zhpmv_inplace_batched_small_kernel_driver< 2>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  3: arginfo = zhpmv_inplace_batched_small_kernel_driver< 3>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  4: arginfo = zhpmv_inplace_batched_small_kernel_driver< 4>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  5: arginfo = zhpmv_inplace_batched_small_kernel_driver< 5>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  6: arginfo = zhpmv_inplace_batched_small_kernel_driver< 6>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  7: arginfo = zhpmv_inplace_batched_small_kernel_driver< 7>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  8: arginfo = zhpmv_inplace_batched_small_kernel_driver< 8>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case  9: arginfo = zhpmv_inplace_batched_small_kernel_driver< 9>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 10: arginfo = zhpmv_inplace_batched_small_kernel_driver<10>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 11: arginfo = zhpmv_inplace_batched_small_kernel_driver<11>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 12: arginfo = zhpmv_inplace_batched_small_kernel_driver<12>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 13: arginfo = zhpmv_inplace_batched_small_kernel_driver<13>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 14: arginfo = zhpmv_inplace_batched_small_kernel_driver<14>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 15: arginfo = zhpmv_inplace_batched_small_kernel_driver<15>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 16: arginfo = zhpmv_inplace_batched_small_kernel_driver<16>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 17: arginfo = zhpmv_inplace_batched_small_kernel_driver<17>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 18: arginfo = zhpmv_inplace_batched_small_kernel_driver<18>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 19: arginfo = zhpmv_inplace_batched_small_kernel_driver<19>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 20: arginfo = zhpmv_inplace_batched_small_kernel_driver<20>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 21: arginfo = zhpmv_inplace_batched_small_kernel_driver<21>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 22: arginfo = zhpmv_inplace_batched_small_kernel_driver<22>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 23: arginfo = zhpmv_inplace_batched_small_kernel_driver<23>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 24: arginfo = zhpmv_inplace_batched_small_kernel_driver<24>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 25: arginfo = zhpmv_inplace_batched_small_kernel_driver<25>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 26: arginfo = zhpmv_inplace_batched_small_kernel_driver<26>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 27: arginfo = zhpmv_inplace_batched_small_kernel_driver<27>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 28: arginfo = zhpmv_inplace_batched_small_kernel_driver<28>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 29: arginfo = zhpmv_inplace_batched_small_kernel_driver<29>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 30: arginfo = zhpmv_inplace_batched_small_kernel_driver<30>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 31: arginfo = zhpmv_inplace_batched_small_kernel_driver<31>(dAP_array, dX_array, incx, batchCount, queue ); break;
        case 32: arginfo = zhpmv_inplace_batched_small_kernel_driver<32>(dAP_array, dX_array, incx, batchCount, queue ); break;
        default:;
    }
}
