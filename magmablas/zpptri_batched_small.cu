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
#define  sA(i, j)  sA[N*(j) - (j)*((j)+1)/2 + (i)]
#define sAi(i, j) sAi[N*(j) - (j)*((j)+1)/2 + (i)]
//#define sAi(i, j) sAi[N*(j) - (j)*((j)+1)/2 + (i)]

#define ZPPTRI_KERNEL_MAX_THREADS (64)

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
template<typename T>
__device__ void print_matrix_packed_lower(
                const char* msg,
                int N, T* sA,
                int tx, int ty, int tz,
                int bx, int by, int bz)
{
#if defined(PRECISION_d) && defined(DBG)
    __syncthreads();
    if(threadIdx.x == tx && threadIdx.y == ty && threadIdx.z == tz &&
       blockIdx.x  == bx && blockIdx.y  == by && blockIdx.z  == bz) {
        printf("%s = [ \n", msg);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i >= j)
                    printf("%8.4f  ", (double)(sA(i,j)));
                else
                   printf("%8.4f  ", (double)(0.));
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
template<int N>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(N)
#endif
void
zpptri_lower_batched_small_kernel(
        magmaDoubleComplex** dAP_array,
        int batchCount, magma_int_t *info_array)
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_N         = ( sizeA / N ) * N;
    constexpr int sizeA_aligned   = ((sizeA+alignment-1)/alignment) * alignment;

    const int tx      = threadIdx.x;
    const int batchid = blockIdx.x;

    magmaDoubleComplex* dA  = dAP_array[batchid];
    magmaDoubleComplex *sA  = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sAi = sA + sizeA_aligned;

    // read A
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N){
         sA[ i + tx ] = dA[ i + tx ];
        sAi[ i + tx ] = MAGMA_Z_ZERO;
    }

    if(tx < (sizeA - sizeA_N)) {
         sA[sizeA_N + tx] = dA[sizeA_N + tx];
        sAi[sizeA_N + tx] = MAGMA_Z_ZERO;
    }
    __syncthreads();

    //print_matrix_packed_lower("sA", N, sA, 0, 0, 0, 0, 0, 0);

    // compute 1 / diagonal(sA)
     sA(tx, tx) = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(tx,tx));
    sAi(tx, tx) = MAGMA_Z_ONE;
    __syncthreads();

    //print_matrix_packed_lower("sAi", N, sAi, 0, 0, 0, 0, 0, 0);

    // Solving L L^T x = I (I)
    // First, solve L y = I for y
    // For now, we do not take full advantage of the identity matrix being the RHS
    #pragma unroll
    for(int i = 0; i < N; i++) {
        if(tx <= i) {
            sAi(i,tx) = sAi(i,tx) * sA(i,i);
            #pragma unroll
            for(int j = i+1; j < N; j++) {
                sAi(j,tx) -= sAi(i,tx) * sA(j,i);
            }
        }
        //print_matrix_packed_lower("sTmp", N, sAi, 0, 0, 0, 0, 0, 0);
    }

    //print_matrix_packed_lower("sAi-1", N, sAi, 0, 0, 0, 0, 0, 0);

    // Second, solve L^T x = y for x
    #pragma unroll
    for(int i = N-1; i >= 0; i--) {
        if( tx <= i ) {
            sAi(i,tx) *= MAGMA_Z_CONJ(sA(i,i));
        }

        #pragma unroll
        for(int j = i-1; j >= 0; j--) {
            if(tx <= j) {
                sAi(j,tx) -= sAi(i, tx) * MAGMA_Z_CONJ( sA(i,j) );
            }
        }
    }
    __syncthreads();

    //print_matrix_packed_lower("sAi-2", N, sAi, 0, 0, 0, 0, 0, 0);

    // overwrite A
    #pragma unroll
    for(int i = 0; i < sizeA_N; i+=N){
        dA[ i + tx ] = sAi[ i + tx ];
    }

    if(tx < (sizeA - sizeA_N)) {
        dA[sizeA_N + tx] = sAi[sizeA_N + tx];
    }

}

////////////////////////////////////////////////////////////////////////////////
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(ZPPTRI_KERNEL_MAX_THREADS)
#endif
void
zpptri_lower_batched_small_kernel_n(
        int n, magmaDoubleComplex** dAP_array,
        int batchCount, magma_int_t *info_array)
{

    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);

    const int N             = n;    // just for the macro expansion of 'sA' and 'sAi'
    const int sizeA         = (n+1)*n/2;
    const int sizeA_aligned = ((sizeA+alignment-1)/alignment) * alignment;

    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int batchid = blockIdx.x;

    magmaDoubleComplex* dA  = dAP_array[batchid];
    magmaDoubleComplex *sA  = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sAi = sA + sizeA_aligned;

    // read A
    for(int i = tx; i < sizeA; i+=ntx) {
         sA[ i ] = dA[ i ];
        sAi[ i ] = MAGMA_Z_ZERO;
    }
    __syncthreads();

    //print_matrix_packed_lower("sA", N, sA, 0, 0, 0, 0, 0, 0);

    // compute 1 / diagonal(sA)
     sA(tx, tx) = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(tx,tx));
    sAi(tx, tx) = MAGMA_Z_ONE;
    __syncthreads();

    //print_matrix_packed_lower("sAi", N, sAi, 0, 0, 0, 0, 0, 0);

    // Solving L L^T x = I (I)
    // First, solve L y = I for y
    for(int i = 0; i < n; i++) {
        if(tx <= i) {
            sAi(i,tx) = sAi(i,tx) * sA(i,i);
            for(int j = i+1; j < n; j++) {
                sAi(j,tx) -= sAi(i,tx) * sA(j,i);
            }
        }
        //print_matrix_packed_lower("sTmp", N, sAi, 0, 0, 0, 0, 0, 0);
    }

    //print_matrix_packed_lower("sAi-1", N, sAi, 0, 0, 0, 0, 0, 0);

    // Second, solve L^T x = y for x
    for(int i = n-1; i >= 0; i--) {
        if( tx <= i ) {
            sAi(i,tx) *= MAGMA_Z_CONJ(sA(i,i));
        }

        for(int j = i-1; j >= 0; j--) {
            if(tx <= j) {
                sAi(j,tx) -= sAi(i, tx) * MAGMA_Z_CONJ( sA(i,j) );
            }
        }
    }
    __syncthreads();

    //print_matrix_packed_lower("sAi-2", N, sAi, 0, 0, 0, 0, 0, 0);

    // overwrite A
    for(int i = tx; i < sizeA; i+=ntx){
        dA[ i ] = sAi[ i ];
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int N>
magma_int_t
zpptri_lower_batched_small_kernel_driver(
    magmaDoubleComplex** dAP_array,
    int batchCount, magma_int_t *info_array, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if ( batchCount < 0 )
        arginfo = -2;

    if (arginfo != 0) {
        return arginfo;
    }

    if( N == 0 || batchCount == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    magma_int_t ntcol = 1; //min(1, 64/N);
    #else
    magma_int_t ntcol = 1; //min(1, 32/N);
    #endif

    // constants
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeA           = (N+1)*N/2;
    constexpr int sizeA_aligned   = ((sizeA+alignment-1)/alignment) * alignment;

    // configure shared memory
    magma_int_t shmem = 0;
    shmem += 2 * ntcol * sizeA_aligned * sizeof(magmaDoubleComplex);

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zpptri_lower_batched_small_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }
    else {
        dim3 threads(N, 1, 1);
        dim3 grid(batchCount, 1, 1);
        void *kernel_args[] = {&dAP_array, &batchCount, &info_array};
        cudaError_t e = cudaLaunchKernel((void*)zpptri_lower_batched_small_kernel<N>, grid, threads, kernel_args, shmem, queue->cuda_stream());
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
magma_zpptri_batched_small(
    magma_int_t n, magmaDoubleComplex** dAP_array,
    magma_int_t batchCount, magma_int_t *info_array,
    magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if(n < 0 || n > 64)
        arginfo = -1;
    else if ( batchCount < 0 )
        arginfo = -3;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || batchCount == 0 ) return 0;

    switch(n){
        case  1: arginfo = zpptri_lower_batched_small_kernel_driver< 1>(dAP_array, batchCount, info_array, queue ); break;
        case  2: arginfo = zpptri_lower_batched_small_kernel_driver< 2>(dAP_array, batchCount, info_array, queue ); break;
        case  3: arginfo = zpptri_lower_batched_small_kernel_driver< 3>(dAP_array, batchCount, info_array, queue ); break;
        case  4: arginfo = zpptri_lower_batched_small_kernel_driver< 4>(dAP_array, batchCount, info_array, queue ); break;
        case  5: arginfo = zpptri_lower_batched_small_kernel_driver< 5>(dAP_array, batchCount, info_array, queue ); break;
        case  6: arginfo = zpptri_lower_batched_small_kernel_driver< 6>(dAP_array, batchCount, info_array, queue ); break;
        case  7: arginfo = zpptri_lower_batched_small_kernel_driver< 7>(dAP_array, batchCount, info_array, queue ); break;
        case  8: arginfo = zpptri_lower_batched_small_kernel_driver< 8>(dAP_array, batchCount, info_array, queue ); break;
        case  9: arginfo = zpptri_lower_batched_small_kernel_driver< 9>(dAP_array, batchCount, info_array, queue ); break;
        case 10: arginfo = zpptri_lower_batched_small_kernel_driver<10>(dAP_array, batchCount, info_array, queue ); break;
        case 11: arginfo = zpptri_lower_batched_small_kernel_driver<11>(dAP_array, batchCount, info_array, queue ); break;
        case 12: arginfo = zpptri_lower_batched_small_kernel_driver<12>(dAP_array, batchCount, info_array, queue ); break;
        case 13: arginfo = zpptri_lower_batched_small_kernel_driver<13>(dAP_array, batchCount, info_array, queue ); break;
        case 14: arginfo = zpptri_lower_batched_small_kernel_driver<14>(dAP_array, batchCount, info_array, queue ); break;
        case 15: arginfo = zpptri_lower_batched_small_kernel_driver<15>(dAP_array, batchCount, info_array, queue ); break;
        case 16: arginfo = zpptri_lower_batched_small_kernel_driver<16>(dAP_array, batchCount, info_array, queue ); break;
        case 17: arginfo = zpptri_lower_batched_small_kernel_driver<17>(dAP_array, batchCount, info_array, queue ); break;
        case 18: arginfo = zpptri_lower_batched_small_kernel_driver<18>(dAP_array, batchCount, info_array, queue ); break;
        case 19: arginfo = zpptri_lower_batched_small_kernel_driver<19>(dAP_array, batchCount, info_array, queue ); break;
        case 20: arginfo = zpptri_lower_batched_small_kernel_driver<20>(dAP_array, batchCount, info_array, queue ); break;
        case 21: arginfo = zpptri_lower_batched_small_kernel_driver<21>(dAP_array, batchCount, info_array, queue ); break;
        case 22: arginfo = zpptri_lower_batched_small_kernel_driver<22>(dAP_array, batchCount, info_array, queue ); break;
        case 23: arginfo = zpptri_lower_batched_small_kernel_driver<23>(dAP_array, batchCount, info_array, queue ); break;
        case 24: arginfo = zpptri_lower_batched_small_kernel_driver<24>(dAP_array, batchCount, info_array, queue ); break;
        case 25: arginfo = zpptri_lower_batched_small_kernel_driver<25>(dAP_array, batchCount, info_array, queue ); break;
        case 26: arginfo = zpptri_lower_batched_small_kernel_driver<26>(dAP_array, batchCount, info_array, queue ); break;
        case 27: arginfo = zpptri_lower_batched_small_kernel_driver<27>(dAP_array, batchCount, info_array, queue ); break;
        case 28: arginfo = zpptri_lower_batched_small_kernel_driver<28>(dAP_array, batchCount, info_array, queue ); break;
        case 29: arginfo = zpptri_lower_batched_small_kernel_driver<29>(dAP_array, batchCount, info_array, queue ); break;
        case 30: arginfo = zpptri_lower_batched_small_kernel_driver<30>(dAP_array, batchCount, info_array, queue ); break;
        case 31: arginfo = zpptri_lower_batched_small_kernel_driver<31>(dAP_array, batchCount, info_array, queue ); break;
        case 32: arginfo = zpptri_lower_batched_small_kernel_driver<32>(dAP_array, batchCount, info_array, queue ); break;
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
        const int sizeA_aligned   = magma_roundup(sizeA, alignment);

        // configure shared memory
        magma_int_t shmem = 0;
        shmem += 2 * sizeA_aligned * sizeof(magmaDoubleComplex);

        int shmem_max = 0;
        #if CUDA_VERSION >= 9000
        cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (shmem <= shmem_max) {
            cudaFuncSetAttribute(zpptri_lower_batched_small_kernel_n, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        }
        #else
        cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
        #endif    // CUDA_VERSION >= 9000

        if ( shmem > shmem_max ) {
            arginfo = -100;;
        }
        else {
            // configure grid and threads
            dim3 threads(n, 1, 1);
            dim3 grid(batchCount, 1, 1);
            void *kernel_args[] = {&n, &dAP_array, &batchCount, &info_array};
            cudaError_t e = cudaLaunchKernel((void*)zpptri_lower_batched_small_kernel_n, grid, threads, kernel_args, shmem, queue->cuda_stream());
            if( e != cudaSuccess ) {
                //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
                arginfo = -100;
            }
        }
    }

    return arginfo;
}
