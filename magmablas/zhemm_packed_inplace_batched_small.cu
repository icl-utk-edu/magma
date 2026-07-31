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

// leading dimension in shared memory
#define SLDA(n)    ( ((n+1)%4) == 0 ? (n) : (n+1) )

// regular access for dB/sB
#define dB(i, j) dB[(j) * lddb + (i)]
#define sA(i, j) sA[(j) * slda + (i)]
#define sB(i, j) sB[(j) * sldb + (i)]

#define ZPPINV_APPLY_KERNEL_MAX_THREADS (64)

////////////////////////////////////////////////////////////////////////////////
// Computes B = A * B
// A is MxM (Hermitian a matrix in packed format), B is MxN (column-major)
// M is the size of the packed matrix
// M is also equivalent to BLOCK_X
// NTHREADS = Tx * TY, must be >= M
template<int M, int BLOCK_Y, int TX, int TY>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(TX*TY)
#endif
void
zhemm_packed_inplace_batched_small_kernel(
        int n,
        magmaDoubleComplex** dAPinv_array,
        magmaDoubleComplex** dB_array, int lddb,
        int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int slda            = SLDA(M);
    constexpr int sldb            = SLDA(M);
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = slda*M;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;
    constexpr int BLOCK_X         = M;
    constexpr int BLOCK_K         = M;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int bx  = blockIdx.x;
    const int batchid = blockIdx.z;

    magmaDoubleComplex* dA = dAPinv_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    // advance dB
    dB += bx * BLOCK_Y * lddb;

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB = sA + sizeA_aligned;

    const int local_nrhs = min(BLOCK_Y, n - bx * BLOCK_Y);

    // reg. accumulator
    magmaDoubleComplex rC[BLOCK_Y/TY][BLOCK_X/TX] = {MAGMA_Z_ZERO};

    // read A (packed in glmem to full in shmem)
    // for now use the first M threads to read
    if(tid < M) {
        #pragma unroll
        for(int j = 0; j < M; j++) {
            int offset = M*j - (j*(j+1)/2) + tid;
            sA(tid, j) = (tid >= j) ? dA[ offset ] : MAGMA_Z_ZERO;
        }
    }
    __syncthreads();

    // read B
    #pragma unroll
    for(int j = 0; j < BLOCK_Y; j+=TY)
        #pragma unroll
        for(int i = 0; i < BLOCK_X; i+=TX) {
            sB(i+tx, j+ty) = ( (i+tx) < M && (j + ty) < local_nrhs ) ? dB(i+tx, j+ty) : MAGMA_Z_ZERO;
        }

    // make A Hermitian in shared memory
    if(tid < M) {
        #pragma unroll
        for(int j = 0; j < M; j++) {
            sA(tid, j) = (tid < j) ? MAGMA_Z_CONJ( sA(j,tid) ) : sA(tid, j);
        }
    }
    __syncthreads();

    //print_memory( "sA2", M, M,          sA, slda,  0, 0, 0, 0, 0, 0);
    //print_memory( "sB", M, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    // multiply
    #pragma unroll
    for(int k = 0; k < BLOCK_K; k++) {
       #pragma unroll
       for(int j = 0; j < BLOCK_Y; j+= TY) {
            #pragma unroll
            for(int i = 0; i < BLOCK_X; i+=TX) {
                rC[j/TY][i/TX] += sA(i+tx,k) * sB(k,j+ty);
            }
        }
    }

    //print_memory( "sB", M, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    // update B
    #pragma unroll
    for(int j = 0; j < BLOCK_Y; j+=TY) {
        #pragma unroll
        for(int i = 0; i < BLOCK_X; i+=TX) {
            if( (i+tx) < M && (j + ty) < local_nrhs) {
                dB(i+tx, j+ty) = rC[j/TY][i/TX];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int M, int BLOCK_Y, int TX, int TY>
magma_int_t
zhemm_packed_inplace_batched_small_kernel_driver(
    int n,
    magmaDoubleComplex** dAPinv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    int batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if(n < 0)
        arginfo = -1;
    else if( lddb < M )
        arginfo = -4;
    else if ( batchCount < 0 )
        arginfo = -5;

    if (arginfo != 0) {
        return arginfo;
    }

    if( M == 0 || n == 0 || batchCount == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    magma_int_t ntcol = 1;//min(1, 64 / M);
    #else
    magma_int_t ntcol = 1;//min(1, 32 / M);
    #endif

    // assertions
    static_assert(TX*TY >= M);
    static_assert(M % TX == 0);
    static_assert(BLOCK_Y % TY == 0);

    // constants
    constexpr int slda            = SLDA(M);
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = slda * M;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;


    constexpr int sldb          = SLDA(M);
    constexpr int sizeB         = sldb * BLOCK_Y;
    constexpr int sizeB_aligned = ( (sizeB + alignment - 1) / alignment) * alignment;

    // configure shared memory
    magma_int_t shmem = 0;
    shmem += ntcol * sizeA_aligned * sizeof(magmaDoubleComplex);
    shmem += ntcol * sizeB_aligned * sizeof(magmaDoubleComplex);

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zhemm_packed_inplace_batched_small_kernel<M, BLOCK_Y, TX, TY>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }
    else {
        magma_int_t gridx = magma_ceildiv(n, BLOCK_Y);
        dim3 threads(TX, TY, 1);

        magma_int_t max_batchCount = queue->get_maxBatch();
        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid(gridx, 1, ibatch);
            magmaDoubleComplex** dAPinv_array_ = dAPinv_array + i;
            magmaDoubleComplex** dB_array_     = dB_array     + i;
            void *kernel_args[] = {&n, &dAPinv_array_, &dB_array_, &lddb, &ibatch};
            cudaError_t e = cudaLaunchKernel((void*)zhemm_packed_inplace_batched_small_kernel<M, BLOCK_Y, TX, TY>, grid, threads, kernel_args, shmem, queue->cuda_stream());
            if( e != cudaSuccess ) {
                //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
                arginfo = -100;
            }
        }
    }
    return arginfo;
}
/***************************************************************************//**
    Purpose
    -------
    PPINV_APPLY multilpes the inverse of a Hermitian matrix 'A' by a dense matrix B
    The matrix B is overwritten with the product

    This is a batched version that factors batchCount M-by-M matrices in parallel.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  M >= 0.

    @param[in,out]
    dAPinv_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,M).
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
magma_zhemm_packed_inplace_batched_small(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dAPinv_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if( side != MagmaLeft ) {
        arginfo = -1;
        printf("Only side = MagmaLeft is currently supported\n");
    }
    else if( uplo != MagmaLower ) {
        arginfo = -2;
        printf("Only uplo = MagmaLower is currently supported\n");
    }
    else if( m < 0 || m > 64 )
        arginfo = -3;
    else if( n < 0 )
        arginfo = -4;
    else if( lddb < max(1,m) )
        arginfo = -7;
    else if ( batchCount < 0 )
        arginfo = -8;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    if( m == 0 || n == 0 || batchCount == 0 ) return;

    switch(m){
        #ifdef TUNE_PPINV_APPLY
        case TUNE_PPINV_N: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< PPINV_BX, PPINV_BY, PPINV_TX, PPINV_TY>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        #else
        case  1: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 1, 32,  1,  1>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  2: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 2, 32,  2, 16>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  3: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 3, 30,  3, 10>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  4: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 4, 32,  4,  8>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  5: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 5, 30,  5,  6>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  6: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 6, 30,  6,  5>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  7: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 7, 32,  7,  4>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  8: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 8, 32,  8,  8>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  9: arginfo = zhemm_packed_inplace_batched_small_kernel_driver< 9, 36, 9,   6>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 10: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<10, 30, 10,  3>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 11: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<11, 11, 11, 11>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 12: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<12, 12, 12, 12>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 13: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<13, 13, 13, 13>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 14: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<14, 14, 14, 14>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 15: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<15, 15, 15, 15>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 16: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<16, 16, 16, 16>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 17: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<17, 15, 17, 15>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 18: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<18, 32,  9,  8>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 19: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<19, 19, 19, 19>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 20: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<20, 20, 20, 20>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 21: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<21, 21, 21, 21>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 22: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<22, 22, 22, 22>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 23: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<23, 23, 23, 23>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 24: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<24, 24, 24, 24>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 25: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<25, 25, 25, 25>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 26: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<26, 26, 26, 26>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 27: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<27, 40, 9,  10>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 28: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<28, 28, 28, 28>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 29: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<29, 29, 29, 29>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 30: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<30, 30, 30, 30>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 31: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<31, 31, 31, 31>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 32: arginfo = zhemm_packed_inplace_batched_small_kernel_driver<32, 32, 32,  4>(n, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        #endif
        default:;
    }
}
