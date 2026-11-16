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

// regular access for dB/sB
#define dB(i, j) dB[(j) * lddb + (i)]
#define sB(i, j) sB[(j) * sldb + (i)]

#define ZPPTRS_KERNEL_MAX_THREADS (32)

////////////////////////////////////////////////////////////////////////////////
// N is the size of the packed matrix
// special kernel for nrhs = 1
template<int N, int NTCOL>
__global__
__launch_bounds__(N*NTCOL)
void
zpptrs_1rhs_lower_batched_small_kernel(
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex** dB_array, int lddb,
        int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;
    constexpr int sizeA_N         = ( sizeA / N ) * N;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;

    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dAP_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    magmaDoubleComplex *sA = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB = sA + blockDim.y * sizeA_aligned;

    sA += ty * sizeA_aligned;
    sB += ty * N;

    // read A
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N){
        sA[ i + tx ] = dA[ i + tx ];
    }

    if(tx < (sizeA - sizeA_N)) {
        sA[sizeA_N + tx] = dA[sizeA_N + tx];
    }

    // read B
    sB[ tx ] = dB[ tx ];
    magmablas_syncwarp();

    // compute 1 / diagonal(sA)
    if(tx < N) {
        sA(tx, tx) = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(tx,tx));
    }
    magmablas_syncwarp();

    // forward solve
    #pragma unroll
    for(int i = 0; i < N; i++) {
        magmaDoubleComplex rT1 = sB[i] * sA(i,i);
        magmaDoubleComplex rT2 = (tx == i) ? rT1 : (sB[tx] - rT1*sA(tx,i));
        magmablas_syncwarp();
        sB[tx]                 = (tx >= i) ? rT2 : sB[tx];
        magmablas_syncwarp();
    }

    // backward solve
    #pragma unroll
    for(int i = N-1; i >= 0; i--) {
        magmaDoubleComplex rT1 = sB[i] * MAGMA_Z_CONJ( sA(i,i) );
        magmaDoubleComplex rT2 = (tx == i) ? rT1 : (sB[tx] - rT1*MAGMA_Z_CONJ( sA(i,tx) ));
        magmablas_syncwarp();
        sB[tx]                 = (tx <= i) ? rT2 : sB[tx];
        magmablas_syncwarp();
    }

    // write back to memory
    dB[ tx ] = sB[ tx ];
}


////////////////////////////////////////////////////////////////////////////////
template<int N>
magma_int_t
zpptrs_1rhs_lower_batched_small_kernel_driver(
    magmaDoubleComplex** dAP_array,
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
    constexpr magma_int_t NTCOL = max(1, 64 / N);
    #else
    constexpr magma_int_t NTCOL = max(1, 32 / N);
    #endif

    // constants
    const int alignment_bytes = 128;
    const int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    const int sizeA           = (N+1)*N/2;
    const int sizeA_aligned   = ( (sizeA + alignment - 1) / alignment) * alignment;

    // configure shared memory
    magma_int_t shmem = 0;
    shmem += sizeA_aligned * sizeof(magmaDoubleComplex);
    shmem += N * sizeof(magmaDoubleComplex);
    shmem *= NTCOL;

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zpptrs_1rhs_lower_batched_small_kernel<N, NTCOL>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }
    else {
        dim3 threads(N, NTCOL, 1);
        dim3 grid(magma_ceildiv(batchCount, NTCOL), 1, 1);
        zpptrs_1rhs_lower_batched_small_kernel<N, NTCOL><<<grid, threads, shmem, queue->cuda_stream()>>>(dAP_array, dB_array, lddb, batchCount);
    }
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZPPTRS solves a system of linear equations A*X = B with a Hermitian positive
    definite matrix A in packed storage using the Cholesky
    factorization A = U**H * U or A = L * L**H computed by ZPPTRF.

    ** Only uplo = 'L' is currently supported

    ** This is the batch version of the operation, performing the computation
       on many independent problems having the same dimensions.

    ** The routine currently supports a limited range of matrix sizes
    ** This version supports only a single right hand side

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER
            The order of the matrix A.  0 <= n <= 32.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  0 <= NRHS <= 1.

    @param[in]
    dAP_array    Array of pointers, dimension (batchCount).
             Each (input) COMPLEX*16 array, dimension (n*(n+1)/2)
             The triangular factor U or L from the Cholesky factorization
             A = U**H * U or A = L * L**H, packed columnwise in a linear array.
             The j-th column of U or L is stored in the array AP as follows:
               - if UPLO = 'U', AP(i + (j-1)*j/2) = U(i,j) for 1<=i<=j;
               - if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = L(i,j) for j<=i<=n.

    @param[in,out]
    dB_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX*16 array, dimension (lddb,nrhs)
               - On entry, the right hand side matrix B.
               - On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  lddb >= max(1,n).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.


    @ingroup magma_potrs_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpptrs_1rhs_batched_small(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(n < 0 || n > 32)
        arginfo = -1;
    else if( nrhs > 1 )
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
        case  1: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 1>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  2: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 2>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  3: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 3>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  4: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 4>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  5: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 5>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  6: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 6>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  7: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 7>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  8: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 8>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case  9: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver< 9>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 10: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<10>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 11: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<11>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 12: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<12>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 13: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<13>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 14: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<14>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 15: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<15>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 16: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<16>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 17: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<17>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 18: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<18>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 19: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<19>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 20: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<20>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 21: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<21>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 22: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<22>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 23: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<23>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 24: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<24>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 25: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<25>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 26: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<26>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 27: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<27>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 28: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<28>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 29: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<29>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 30: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<30>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 31: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<31>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        case 32: arginfo = zpptrs_1rhs_lower_batched_small_kernel_driver<32>(dAP_array, dB_array, lddb, batchCount, queue ); break;
        default: arginfo = -100;
    }

    return arginfo;
}
