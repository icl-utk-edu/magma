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
// special kernel for nrhs = 1
template<int N, int NTCOL>
__global__
//#ifdef MAGMA_HAVE_HIP
__launch_bounds__(N*NTCOL)
//#endif
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

    //print_memory( "sA", sizeA, 1, sA, sizeA, 0, 0, 0, 0, 0, 0);
    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

    #if 0
    if( tx == 0 ) {
        // Solving L L^T x = b
        // First, solve L y = b for y
        #pragma unroll
        for(int i = 0; i < N; i++) {
            sB[i] *= sA(i,i);
            #pragma unroll
            for(int j = i+1; j < N; j++) {
                sB[j] -= sB[i] * sA(j,i);
            }
        }

        // Second, solve L^T x = y for x
        #pragma unroll
        for(int i = N-1; i >= 0; i--) {
            sB[i] *= MAGMA_Z_CONJ(sA(i,i));
            #pragma unroll
            for(int j = i-1; j >= 0; j--) {
                sB[j] -= sB[i] * MAGMA_Z_CONJ( sA(i,j) );
            }
        }
    }
    magmablas_syncwarp();
    #else
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
    #endif

    //print_memory( "sB", N, local_nrhs, sB, sldb,  0, 0, 0, 0, 0, 0);

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
magma_zpptrs_1rhs_batched_small(
    magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex** dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(n < 0 || n > 64)
        arginfo = -1;
    else if( nrhs != 1 )
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

            magma_int_t max_batchCount = queue->get_maxBatch();
            for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid(gridx, 1, ibatch);
                magmaDoubleComplex** dAP_array_ = dAP_array + i;
                magmaDoubleComplex** dB_array_  = dB_array  + i;
                void *kernel_args[] = {&n, &nrhs, &nrhs_nb, &dAP_array_, &dB_array_, &lddb, &batchCount};
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
