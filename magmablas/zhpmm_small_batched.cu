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
#define dC(i, j) dC[(j) * lddc + (i)]
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
zhpmm_lower_left_small_batched_kernel(
        int n,
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex** dB_array, int lddb,
        magmaDoubleComplex** dC_array, int lddc,
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

    magmaDoubleComplex* dA = dAP_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magmaDoubleComplex* dC = dC_array[batchid];

    // advance dB
    dB += bx * BLOCK_Y * lddb;
    dC += bx * BLOCK_Y * lddc;

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

    // write C
    #pragma unroll
    for(int j = 0; j < BLOCK_Y; j+=TY) {
        #pragma unroll
        for(int i = 0; i < BLOCK_X; i+=TX) {
            if( (i+tx) < M && (j + ty) < local_nrhs) {
                dC(i+tx, j+ty) = rC[j/TY][i/TX];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int M, int BLOCK_Y, int TX, int TY>
magma_int_t
zhpmm_lower_left_small_batched_kernel_driver(
    int n,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magmaDoubleComplex** dC_array, magma_int_t lddc,
    int batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if(n < 0)
        arginfo = -1;
    else if( lddb < M )
        arginfo = -4;
    else if( lddc < M )
        arginfo = -6;
    else if ( batchCount < 0 )
        arginfo = -7;

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
        cudaFuncSetAttribute(zhpmm_lower_left_small_batched_kernel<M, BLOCK_Y, TX, TY>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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
            magmaDoubleComplex** dAP_array_ = dAP_array + i;
            magmaDoubleComplex** dB_array_  = dB_array  + i;
            magmaDoubleComplex** dC_array_  = dC_array  + i;
            void *kernel_args[] = {&n, &dAP_array_, &dB_array_, &lddb, &dC_array_, &lddc, &ibatch};
            cudaError_t e = cudaLaunchKernel((void*)zhpmm_lower_left_small_batched_kernel<M, BLOCK_Y, TX, TY>, grid, threads, kernel_args, shmem, queue->cuda_stream());
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
    ZHPMM performs the update:
            C = AP * B     side='left', or
            C = B  * AP    side='right' (not currently supported)
    , where AP is a Hermitian positive definite matrix stored in packed
    format, and B is a general matrix that is stored in column-major format.
    The packed format stores either the lower (uplo = 'L') or
    the upper (uplo = 'U') part of AP.

    ** Only side = 'left' is currently supported
    ** Only uplo = 'L'    is currently supported
    ** For small sizes, the update can be done in-place
       (B and C point to the same matrix)

    ** This is the batch version of the operation, performing the computation
       on many independent matrices having the same dimensions.

    ** The routine currently supports a limited range of sizes for the matrix AP

    Arguments
    ---------
    @param[in]
    side    magma_side_t
            On entry, side specifies whether each Hermitian matrix AP
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    B := AP*B
            SIDE = MagmaRight   B := B*AP
      **    Only MagmaLeft is supported.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
      **    Only MagmaLower is supported.

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of each matrix C.
            m >= 0.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of each matrix C.
            n >= 0.

    @param[in]
    dAP_array    Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (m*(m+1)/2)
             On entry, the upper or lower triangle part of each matrix AP,
             packed columnwise in a linear array.
             Unchanged on exit.

    @param[in]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (lddb,n)
               - On entry, the input matrix B.
               - Unchanged on exit
                 (unless dB_array = dC_array, which is permissible for small sizes)

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  lddb >= max(1,m).

    @param[out]
    dC_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (lddc,n)
               - On exit, the updated matrix C.

    @param[in]
    lddc    INTEGER
            The leading dimension of each array C.  lddc >= max(1,m).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_posv_batched
*******************************************************************************/
extern "C" void
magma_zhpmm_small_batched_core(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dAP_array,
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
        case TUNE_PPINV_N: arginfo = zhpmm_lower_left_small_batched_kernel_driver< PPINV_BX, PPINV_BY, PPINV_TX, PPINV_TY>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        #else
        case  1: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 1, 32,  1,  1>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  2: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 2, 32,  2, 16>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  3: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 3, 30,  3, 10>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  4: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 4, 32,  4,  8>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  5: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 5, 30,  5,  6>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  6: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 6, 30,  6,  5>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  7: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 7, 32,  7,  4>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  8: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 8, 32,  8,  8>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  9: arginfo = zhpmm_lower_left_small_batched_kernel_driver< 9, 36, 9,   6>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 10: arginfo = zhpmm_lower_left_small_batched_kernel_driver<10, 30, 10,  3>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 11: arginfo = zhpmm_lower_left_small_batched_kernel_driver<11, 11, 11, 11>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 12: arginfo = zhpmm_lower_left_small_batched_kernel_driver<12, 12, 12, 12>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 13: arginfo = zhpmm_lower_left_small_batched_kernel_driver<13, 13, 13, 13>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 14: arginfo = zhpmm_lower_left_small_batched_kernel_driver<14, 14, 14, 14>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 15: arginfo = zhpmm_lower_left_small_batched_kernel_driver<15, 15, 15, 15>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 16: arginfo = zhpmm_lower_left_small_batched_kernel_driver<16, 16, 16, 16>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 17: arginfo = zhpmm_lower_left_small_batched_kernel_driver<17, 15, 17, 15>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 18: arginfo = zhpmm_lower_left_small_batched_kernel_driver<18, 32,  9,  8>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 19: arginfo = zhpmm_lower_left_small_batched_kernel_driver<19, 19, 19, 19>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 20: arginfo = zhpmm_lower_left_small_batched_kernel_driver<20, 20, 20, 20>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 21: arginfo = zhpmm_lower_left_small_batched_kernel_driver<21, 21, 21, 21>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 22: arginfo = zhpmm_lower_left_small_batched_kernel_driver<22, 22, 22, 22>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 23: arginfo = zhpmm_lower_left_small_batched_kernel_driver<23, 23, 23, 23>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 24: arginfo = zhpmm_lower_left_small_batched_kernel_driver<24, 24, 24, 24>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 25: arginfo = zhpmm_lower_left_small_batched_kernel_driver<25, 25, 25, 25>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 26: arginfo = zhpmm_lower_left_small_batched_kernel_driver<26, 26, 26, 26>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 27: arginfo = zhpmm_lower_left_small_batched_kernel_driver<27, 40, 9,  10>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 28: arginfo = zhpmm_lower_left_small_batched_kernel_driver<28, 28, 28, 28>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 29: arginfo = zhpmm_lower_left_small_batched_kernel_driver<29, 29, 29, 29>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 30: arginfo = zhpmm_lower_left_small_batched_kernel_driver<30, 30, 30, 30>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 31: arginfo = zhpmm_lower_left_small_batched_kernel_driver<31, 31, 31, 31>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 32: arginfo = zhpmm_lower_left_small_batched_kernel_driver<32, 32, 32,  4>(n, dAP_array, dB_array, lddb, batchCount, queue ); break;
        #endif
        default:;
    }
}

/***************************************************************************//**
    Purpose
    -------
    ZHPMM performs the in-place update:
            B = AP * B     side='left', or
            B = B  * AP    side='right'
    , where AP is a Hermitian positive definite matrix stored in packed
    format, and B is a general matrix that is stored in column-major format.
    The packed format stores either the lower (uplo = 'L') or
    the upper (uplo = 'U') part of AP.

    ** Only side = 'left' is currently supported
    ** Only uplo = 'L'    is currently supported

    ** This is the batch version of the operation, performing the computation
       on many independent matrices having the same dimensions.

    ** The routine currently supports a limited range of sizes for the matrix AP

    Arguments
    ---------
    @param[in]
    side    magma_side_t
            On entry, side specifies whether each Hermitian matrix AP
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    B := AP*B
            SIDE = MagmaRight   B := B*AP
      **    Only MagmaLeft is supported.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
      **    Only MagmaLower is supported.

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of each matrix B.
            m >= 0.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of each matrix AP.
            n >= 0.

    @param[in]
    dAP_array    Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (n*(n+1)/2)
             On entry, the upper or lower triangle part of each matrix AP,
             packed columnwise in a linear array.

             Unchanged nn exit.

    @param[in,out]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (lddb,n)
               - On entry, the input matrix B.
               - On exit, the updated matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  lddb >= max(1,n).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_posv_batched
*******************************************************************************/
extern "C" void
magma_zhpmm_small_batched(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dAP_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magmaDoubleComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    if ( side != MagmaLeft  ) {
        arginfo = -1;
        printf("Only side = MagmaLeft is currently supported\n");
    }
    else if ( uplo != MagmaLower ) {
        arginfo = -2;
        printf("Only uplo = MagmaLower is currently supported\n");
    }
    else if ( m < 0 || m > 32 ) {
        arginfo = -3;
    }
    else if ( n < 0 ) {
        arginfo = -4;
    }
    else if ( lddb < max(1,m) ) {
        arginfo = -7;
    }
    else if ( lddc < max(1,m) ) {
        arginfo = -9;
    }
    else if ( batchCount < 0 )
        arginfo = -10;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    /* Quick return if possible */
    if ( m == 0 || n == 0 || batchCount == 0 ) return;

    // special case
    if( side == MagmaLeft && m <= 32 && n == 1 ) {
        magma_zhpmv_small_batched( uplo, m, dAP_array, dB_array, 1, dC_array, 1, batchCount, queue );
    }
    else {
        magma_zhpmm_small_batched_core(side, uplo, m, n, dAP_array, dB_array, lddb, batchCount, queue );
    }
}

