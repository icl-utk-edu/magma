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
// A is NxN (the inverse of a matrix in packed format), N is Nxnrhs (column-major)
// N is the size of the packed matrix
// N is also equivalent to BLOCK_X
// NTHREADS = Tx * TY, must be >= N
template<int N, int BLOCK_Y, int TX, int TY>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(TX*TY)
#endif
void
zppinv_apply_lower_batched_small_kernel(
        int nrhs,
        magmaDoubleComplex** dAPinv_array,
        magmaDoubleComplex** dB_array, int lddb,
        int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int slda            = SLDA(N);
    constexpr int sldb            = SLDA(N);
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = slda*N;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;
    constexpr int BLOCK_X         = N;
    constexpr int BLOCK_K         = N;

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

    const int local_nrhs = min(BLOCK_Y, nrhs - bx * BLOCK_Y);

    // reg. accumulator
    magmaDoubleComplex rC[BLOCK_Y/TY][BLOCK_X/TX] = {MAGMA_Z_ZERO};

    // read A (packed in glmem to full in shmem)
    // for now use the first N threads to read
    if(tid < N) {
        #pragma unroll
        for(int j = 0; j < N; j++) {
            int offset = N*j - (j*(j+1)/2) + tid;
            sA(tid, j) = (tid >= j) ? dA[ offset ] : MAGMA_Z_ZERO;
        }
    }
    __syncthreads();

    // read B
    #pragma unroll
    for(int j = 0; j < BLOCK_Y; j+=TY)
        #pragma unroll
        for(int i = 0; i < BLOCK_X; i+=TX) {
            sB(i+tx, j+ty) = ( (i+tx) < N && (j + ty) < local_nrhs ) ? dB(i+tx, j+ty) : MAGMA_Z_ZERO;
        }

    // make A Hermitian in shared memory
    if(tid < N) {
        #pragma unroll
        for(int j = 0; j < N; j++) {
            sA(tid, j) = (tid < j) ? MAGMA_Z_CONJ( sA(j,tid) ) : sA(tid, j);
        }
    }
    __syncthreads();

    //print_memory( "sA2", N, N,          sA, slda,  0, 0, 0, 0, 0, 0);
    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

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

    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    // update B
    #pragma unroll
    for(int j = 0; j < BLOCK_Y; j+=TY) {
        #pragma unroll
        for(int i = 0; i < BLOCK_X; i+=TX) {
            if( (i+tx) < N && (j + ty) < local_nrhs) {
                dB(i+tx, j+ty) = rC[j/TY][i/TX];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int N, int BLOCK_Y, int TX, int TY>
magma_int_t
zppinv_apply_lower_batched_small_kernel_driver(
    int nrhs,
    magmaDoubleComplex** dAPinv_array,
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
    magma_int_t ntcol = 1;//min(1, 64 / N);
    #else
    magma_int_t ntcol = 1;//min(1, 32 / N);
    #endif

    // assertions
    static_assert(TX*TY >= N);
    static_assert(N % TX == 0);
    static_assert(BLOCK_Y % TY == 0);

    // constants
    constexpr int slda            = SLDA(N);
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = slda * N;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;


    constexpr int sldb          = SLDA(N);
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
        cudaFuncSetAttribute(zppinv_apply_lower_batched_small_kernel<N, BLOCK_Y, TX, TY>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }
    else {
        magma_int_t gridx = magma_ceildiv(nrhs, BLOCK_Y);
        dim3 threads(TX, TY, 1);

        magma_int_t max_batchCount = queue->get_maxBatch();
        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid(gridx, 1, ibatch);
            magmaDoubleComplex** dAPinv_array_ = dAPinv_array + i;
            magmaDoubleComplex** dB_array_     = dB_array     + i;
            void *kernel_args[] = {&nrhs, &dAPinv_array_, &dB_array_, &lddb, &ibatch};
            cudaError_t e = cudaLaunchKernel((void*)zppinv_apply_lower_batched_small_kernel<N, BLOCK_Y, TX, TY>, grid, threads, kernel_args, shmem, queue->cuda_stream());
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
extern "C" magma_int_t
magma_zppinv_apply_batched_small(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dAPinv_array,
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
        #ifdef TUNE_PPINV_APPLY
        case TUNE_PPINV_N: arginfo = zppinv_apply_lower_batched_small_kernel_driver< PPINV_BX, PPINV_BY, PPINV_TX, PPINV_TY>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        #else
        case  1: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 1, 32,  1,  1>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  2: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 2, 32,  2, 16>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  3: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 3, 30,  3, 10>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  4: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 4, 32,  4,  8>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  5: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 5, 30,  5,  6>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  6: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 6, 30,  6,  5>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  7: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 7, 32,  7,  4>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  8: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 8, 32,  8,  8>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case  9: arginfo = zppinv_apply_lower_batched_small_kernel_driver< 9, 36, 9,   6>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 10: arginfo = zppinv_apply_lower_batched_small_kernel_driver<10, 30, 10,  3>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 11: arginfo = zppinv_apply_lower_batched_small_kernel_driver<11, 11, 11, 11>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 12: arginfo = zppinv_apply_lower_batched_small_kernel_driver<12, 12, 12, 12>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 13: arginfo = zppinv_apply_lower_batched_small_kernel_driver<13, 13, 13, 13>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 14: arginfo = zppinv_apply_lower_batched_small_kernel_driver<14, 14, 14, 14>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 15: arginfo = zppinv_apply_lower_batched_small_kernel_driver<15, 15, 15, 15>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 16: arginfo = zppinv_apply_lower_batched_small_kernel_driver<16, 16, 16, 16>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 17: arginfo = zppinv_apply_lower_batched_small_kernel_driver<17, 15, 17, 15>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 18: arginfo = zppinv_apply_lower_batched_small_kernel_driver<18, 32,  9,  8>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 19: arginfo = zppinv_apply_lower_batched_small_kernel_driver<19, 19, 19, 19>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 20: arginfo = zppinv_apply_lower_batched_small_kernel_driver<20, 20, 20, 20>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 21: arginfo = zppinv_apply_lower_batched_small_kernel_driver<21, 21, 21, 21>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 22: arginfo = zppinv_apply_lower_batched_small_kernel_driver<22, 22, 22, 22>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 23: arginfo = zppinv_apply_lower_batched_small_kernel_driver<23, 23, 23, 23>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 24: arginfo = zppinv_apply_lower_batched_small_kernel_driver<24, 24, 24, 24>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 25: arginfo = zppinv_apply_lower_batched_small_kernel_driver<25, 25, 25, 25>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 26: arginfo = zppinv_apply_lower_batched_small_kernel_driver<26, 26, 26, 26>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 27: arginfo = zppinv_apply_lower_batched_small_kernel_driver<27, 40, 9,  10>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 28: arginfo = zppinv_apply_lower_batched_small_kernel_driver<28, 28, 28, 28>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 29: arginfo = zppinv_apply_lower_batched_small_kernel_driver<29, 29, 29, 29>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 30: arginfo = zppinv_apply_lower_batched_small_kernel_driver<30, 30, 30, 30>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 31: arginfo = zppinv_apply_lower_batched_small_kernel_driver<31, 31, 31, 31>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        case 32: arginfo = zppinv_apply_lower_batched_small_kernel_driver<32, 32, 32,  4>(nrhs, dAPinv_array, dB_array, lddb, batchCount, queue ); break;
        #endif
        default: arginfo = -100;
    }

    #if 0
    if(arginfo != 0) {
        arginfo = 0;

        magma_device_t device;
        magma_getdevice( &device );

        // constants
        const int alignment_bytes = 128;
        const int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
        const int sizeA           = (n+1)*n/2;
        const int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;

        int BLOCK_Y = n;  // BLOCK_Y is a blocking size for right hand sides (NRHS >= N)

        const int sldb          = SLDA(n);
        const int sizeB         = sldb * BLOCK_Y;
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
            magma_int_t gridx = magma_ceildiv(nrhs, BLOCK_Y);
            dim3 threads(BLOCK_Y, 1, 1);

            magma_int_t max_batchCount = queue->get_maxBatch();
            for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid(gridx, 1, ibatch);
                magmaDoubleComplex** dAPinv_array_ = dAPinv_array + i;
                magmaDoubleComplex** dB_array_  = dB_array  + i;
                void *kernel_args[] = {&n, &nrhs, &BLOCK_Y, &dAPinv_array_, &dB_array_, &lddb, &batchCount};
                cudaError_t e = cudaLaunchKernel((void*)zpptrs_lower_batched_small_kernel_n, grid, threads, kernel_args, shmem, queue->cuda_stream());
                if( e != cudaSuccess ) {
                    //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
                    arginfo = -300;
                }
            }
        }
    }
    #endif

    return arginfo;
}
