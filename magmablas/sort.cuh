/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "swap_scalar.cuh"

#define SORT_SM_MAX_THREADS (1024)
#define SORT_SM_MAX_LENGTH  (2 * SORT_SM_MAX_THREADS)

////////////////////////////////////////////////////////////////////////////////
template<typename T>
T magmablas_get_rmin() = delete;

template<>
float magmablas_get_rmin<float>() { return lapackf77_slamch("U"); }

template<>
float magmablas_get_rmax<float>() { return lapackf77_slamch("O"); }

template<>
double magmablas_get_rmin<double>() { return lapackf77_dlamch("U"); }

template<>
double magmablas_get_rmax<double>() { return lapackf77_dlamch("O"); }

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ __inline__ void
magmablas_swap_scalar_device(T& a, T& b) = delete;

////////////////////////////////////////////////////////////////////////////////
// specialization for float
template<>
__device__ __inline__ void
magmablas_swap_scalar_device<float>(float& a, float& b)
{
    magmablas_sswap_scalar_device(a, b);
}

////////////////////////////////////////////////////////////////////////////////
// specialization for double
template<>
__device__ __inline__ void
magmablas_swap_scalar_device<double>(double& a, double& b)
{
    magmablas_dswap_scalar_device(a, b);
}

////////////////////////////////////////////////////////////////////////////////
unsigned int next_pow2_32bit(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return (magma_int_t)n;
}

////////////////////////////////////////////////////////////////////////////////
// NSTEPS = 0.5 * < n rounded up to the next power of 2 >
template<typename T, int NSTEPS>
__device__ static __noinline__ void
min_reduce_key_device(const int n, const int tx, T* x, int* ind)
{
    #pragma unroll
    for(int step = NSTEPS; step > 0; step >>= 1) {
        if ( tx < step && (tx + step) < n ) {
            if ( x[ tx ] > x[tx+step] ) {
                magmablas_iswap_scalar_device(ind[tx], ind[tx+step]);
                magmablas_swap_scalar_device (  x[tx],   x[tx+step]);
            }
        }
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////
// NSTEPS = 0.5 * < n rounded up to the next power of 2 >
template<typename T, int NSTEPS>
__device__ static __noinline__ void
max_reduce_key_device(const int n, const int tx, T* x, int* ind)
{
    #pragma unroll
    for(int step = NSTEPS; step > 0; step >>= 1) {
        if ( tx < step && (tx + step) < n ) {
            if ( x[ tx ] < x[tx+step] ) {
                magmablas_iswap_scalar_device(ind[tx], ind[tx+step]);
                magmablas_swap_scalar_device (  x[tx],   x[tx+step]);
            }
        }
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////
// sort with index in shared memory - ascending order
// NSTEPS = 0.5 * <n rounded up to the next power of 2>
template<typename T, int NSTEPS>
__device__ static __noinline__ void
sort_index_descend(const int n, const int tx, T* sx, int* sindex)
{
    for(int in = 0; in < n-1; in++) {
        min_reduce_key_device<T, NSTEPS>(n-in, tx, sx+in, sindex+in);
    }
}

////////////////////////////////////////////////////////////////////////////////
// sort with index in shared memory - descending order
// NSTEPS = 0.5 * <n rounded up to the next power of 2>
template<typename T, int NSTEPS>
__device__ static __noinline__ void
sort_index_descend(const int n, const int tx, T* sx, int* sindex)
{
    for(int in = 0; in < n-1; in++) {
        max_reduce_key_device<T, NSTEPS>(n-in, tx, sx+in, sindex+in);
    }
}

////////////////////////////////////////////////////////////////////////////////
// sort (with index) in shared memory
// NSTEPS = 0.5 * <n rounded up to the next power of 2>
// Configured with NSTEPS threads
// maximum length = SORT_SM_MAX_LENGTH
// This is a trivial sort, not expected to be fast
// TODO: implement faster algorithms, like radix or bitonic sort
template<typename T, int NSTEPS>
__global__ void
sort_descend_sm_kernel_batched(
    int n,
    T const * const *  dx_array, int incx,
    T** dy_array, int incy,
    magma_int_t** dindex_array)
{
    // Do not use the declaration below if multiple instantiations are in the same file
    //extern __shared__ T* smdata[];
    extern __shared__ __align__(sizeof(T)) unsigned char smdata[];

    const int batchid = blockIdx.x;
    const int tx      = threadIdx.x;
    const int tx2     = tx + NSTEPS;

    T* dx = (T*)dx_array[batchid];
    T* dy = dy_array[batchid];

    magma_int_t* dindex = dindex_array[batchid];

    T*   sx     = reinterpret_cast<T*>( smdata );
    int* sindex = (int*)(sx + n);

    // init shared memory
        sx[ tx ] = dx[ tx*incx ];
    sindex[ tx ] = tx;

    if(tx2 < n) {
            sx[ tx2 ] = dx[ tx2*incx ];
        sindex[ tx2 ] = tx2;
    }
    __syncthreads();

    sort_index_descend<T, NSTEPS>
    (n, tx, sx, sindex);

    // write output
    dy[tx*incy] = sx[tx];
    dindex[tx]  = (magma_int_t)sindex[tx];

    if(tx2 < n) {
        dy[tx2*incy] = sx[tx2];
        dindex[tx2]  = (magma_int_t)sindex[tx2];
    }
}

////////////////////////////////////////////////////////////////////////////////
// sort (with index) for any length
// This is a trivial sort, not expected to be fast
// TODO: implement faster algorithms, like radix or bitonic sort
template<typename T, int DIMX>
__global__ void
sort_descend_kernel_batched(
    int n,
    T const * const *  dx_array, int incx,
    T** dy_array, int incy,
    magma_int_t** dindex_array,
    T min_value)
{
    __shared__ T sB[ DIMX ];

    const int batchid = blockIdx.z;
    const int bx      = blockIdx.x;
    const int tx      = threadIdx.x;
    const int gtx     = bx * blockDim.x + tx;

    T* dx = (T*)dx_array[batchid];
    T* dy = dy_array[batchid];

    magma_int_t* dindex = dindex_array[batchid];

    T rA = (gtx < n) ? dx[gtx * incx] : min_value;
    T rB;
    int position = 0;

    for(int i = 0; i < n; i+=blockDim.x) {
        int length = min(blockDim.x, n-i);

        if(tx < length) {
            sB[ tx ] = dx[(tx+i) * incx];
        }
        __syncthreads();

        for(int j = 0; j < length; j++) {
            rB = sB[ j ];
            position += ((rB > rA) ? 1 : 0);
            position += ((rB == rA && (i+j) < gtx) ? 1 : 0);
            //printf("[%d]: rA = %.4f  -- rB = %.4f -- position = %d\n", gtx, rA, rB, position);
        }
        __syncthreads();
    }

    // write output
    if(gtx < n) {
        dy[position * incy] = rA;
        dindex[position]    = (magma_int_t)gtx;
    }
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
static void
magmablas_sort_descend_sm_batched_kernel_driver(
    magma_int_t n,
    T const * const * dx_array, magma_int_t incx,
    T               **dy_array, magma_int_t incy,
    magma_int_t** dindex_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    if( n <= SORT_SM_MAX_LENGTH) {
        magma_int_t shmem = 0;
        shmem += n * sizeof(T);
        shmem += n * sizeof(int); // not magma_int_t

        magma_int_t nsteps = (magma_int_t)next_pow2_32bit( (unsigned int)n ) >> 1;

        dim3 grid(batchCount, 1, 1);
        dim3 threads(nsteps, 1, 1);

        switch( nsteps ) {
            case    1: sort_descend_sm_kernel_batched<T,    1><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case    2: sort_descend_sm_kernel_batched<T,    2><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case    4: sort_descend_sm_kernel_batched<T,    4><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case    8: sort_descend_sm_kernel_batched<T,    8><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case   16: sort_descend_sm_kernel_batched<T,   16><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case   32: sort_descend_sm_kernel_batched<T,   32><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case   64: sort_descend_sm_kernel_batched<T,   64><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case  128: sort_descend_sm_kernel_batched<T,  128><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case  256: sort_descend_sm_kernel_batched<T,  256><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case  512: sort_descend_sm_kernel_batched<T,  512><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case 1024: sort_descend_sm_kernel_batched<T, 1024><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            default:;
        }
    }
    else {
        constexpr int nthreads = 128;
        const   T min_value    = - magmablas_get_rmax<T>();
        const int blocks       = magma_ceildiv(n, nthreads);
        dim3 threads(nthreads, 1, 1);

        magma_int_t max_batchCount = queue->get_maxBatch();
        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( blocks, 1, ibatch );
            sort_descend_kernel_batched<T, nthreads><<<grid, threads, 0, queue->cuda_stream()>>>
            (n, dx_array+i, incx, dy_array+i, incy, dindex_array+i, min_value);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
static void
magmablas_sort_ascend_sm_batched_kernel_driver(
    magma_int_t n,
    T const * const * dx_array, magma_int_t incx,
    T               **dy_array, magma_int_t incy,
    magma_int_t** dindex_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    if( n <= SORT_SM_MAX_LENGTH) {
        magma_int_t shmem = 0;
        shmem += n * sizeof(T);
        shmem += n * sizeof(int); // not magma_int_t

        magma_int_t nsteps = (magma_int_t)next_pow2_32bit( (unsigned int)n ) >> 1;

        dim3 grid(batchCount, 1, 1);
        dim3 threads(nsteps, 1, 1);

        switch( nsteps ) {
            case    1: sort_ascend_sm_kernel_batched<T,    1><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case    2: sort_ascend_sm_kernel_batched<T,    2><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case    4: sort_ascend_sm_kernel_batched<T,    4><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case    8: sort_ascend_sm_kernel_batched<T,    8><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case   16: sort_ascend_sm_kernel_batched<T,   16><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case   32: sort_ascend_sm_kernel_batched<T,   32><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case   64: sort_ascend_sm_kernel_batched<T,   64><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case  128: sort_ascend_sm_kernel_batched<T,  128><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case  256: sort_ascend_sm_kernel_batched<T,  256><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case  512: sort_ascend_sm_kernel_batched<T,  512><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            case 1024: sort_ascend_sm_kernel_batched<T, 1024><<<grid, threads, shmem, queue->cuda_stream()>>>( n, dx_array, incx, dy_array, incy, dindex_array ); break;
            default:;
        }
    }
    else {
        constexpr int nthreads = 128;
        const   T min_value    = magmablas_get_rmax<T>();
        const int blocks       = magma_ceildiv(n, nthreads);
        dim3 threads(nthreads, 1, 1);

        magma_int_t max_batchCount = queue->get_maxBatch();
        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( blocks, 1, ibatch );
            sort_descend_kernel_batched<T, nthreads><<<grid, threads, 0, queue->cuda_stream()>>>
            (n, dx_array+i, incx, dy_array+i, incy, dindex_array+i, min_value);
        }
    }
}


/***************************************************************************//**
    Purpose
    -------

    magmablas_dsort_batched sorts a batch of independent floating point arrays.
      ** This is an out-of-place sort
      ** The routine also outputs an index array representing the permutation
         done to the original array to produce the output array
      ** Disclaimer: This routine is mainly intended for batch routines implementing
         EVD's and SVD's. Since the overhead of sorting in these solvers is usually
         negligible, the routine provides only trivial implementations with no emphasis
         on performance.

    Arguments
    ----------
    @param[in]
    sort  magma_sort_t
          The direction of sorting (MagmaAscending or MagmaDescending)

    @param[in]
    n     INTEGER
          The length of each array X in the batch

    @param[in]
    dx_array  Array of pointers, dimension(batchCount)
              Each is DOUBLE array of length n representing the an array
              to be sorted.

    @param[in]
    incx  INTEGER
          On entry, incx specifies the increment for the elements of each
          array X. INCX > 0.

    @param[out]
    dy_array  Array of pointers, dimension (batchCount).
              Each is DOUBLE array of length n representing the sorted array Y.

    @param[in]
    incy      INTEGER
              On entry, incy specifies the increment for the elements of each
              sorted array Y. INCY > 0.

    @param[out]
    dindex_array  Array of pointers, dimension (batchCount).
                  Each is an integer array of the sorting indices.

    @param[in]
    batchCount  INTEGER
                The number of arrays to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C"
magma_int_t
magmablas_dsort_batched(
    magma_sort_t sort, magma_int_t n,
    double const * const * dx_array, magma_int_t incx,
    double               **dy_array, magma_int_t incy,
    magma_int_t** dindex_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if(n < 0) {
        arginfo = -1;
    }
    else if(incx < 1) {
        arginfo = -3;
    }
    else if(incy < 1) {
        arginfo = -5;
    }
    else if (batchCount < 0) {
        arginfo = -7;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n < 2 || batchCount == 0) return arginfo;

    if(sort == MagmaDescending) {
        magmablas_sort_descend_sm_batched_kernel_driver<double>(n, dx_array, incx, dy_array, incy, dindex_array, batchCount, queue);
    }
    else {
        magmablas_sort_ascend_sm_batched_kernel_driver<double>(n, dx_array, incx, dy_array, incy, dindex_array, batchCount, queue);
    }

    return arginfo;
}


/***************************************************************************//**
    Purpose
    -------

    magmablas_ssort_batched sorts a batch of independent floating point arrays.
      ** This is an out-of-place sort
      ** The routine also outputs an index array representing the permutation
         done to the original array to produce the output array
      ** Disclaimer: This routine is mainly intended for batch routines implementing
         EVD's and SVD's. Since the overhead of sorting in these solvers is usually
         negligible, the routine provides only trivial implementations with no emphasis
         on performance.

    Arguments
    ----------
    @param[in]
    sort  magma_sort_t
          The direction of sorting (MagmaAscending or MagmaDescending)

    @param[in]
    n     INTEGER
          The length of each array X in the batch

    @param[in]
    dx_array  Array of pointers, dimension(batchCount)
              Each is FLOAT array of length n representing the an array
              to be sorted.

    @param[in]
    incx  INTEGER
          On entry, incx specifies the increment for the elements of each
          array X. INCX > 0.

    @param[out]
    dy_array  Array of pointers, dimension (batchCount).
              Each is FLOAT array of length n representing the sorted array Y.

    @param[in]
    incy      INTEGER
              On entry, incy specifies the increment for the elements of each
              sorted array Y. INCY > 0.

    @param[out]
    dindex_array  Array of pointers, dimension (batchCount).
                  Each is an integer array of the sorting indices.

    @param[in]
    batchCount  INTEGER
                The number of arrays to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C"
magma_int_t
magmablas_ssort_batched(
    magma_sort_t sort_type, magma_int_t n,
    float const * const * dx_array, magma_int_t incx,
    float               **dy_array, magma_int_t incy,
    magma_int_t** dindex_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if(n < 0) {
        arginfo = -1;
    }
    else if(incx < 1) {
        arginfo = -3;
    }
    else if(incy < 1) {
        arginfo = -5;
    }
    else if (batchCount < 0) {
        arginfo = -7;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if(n < 2 || batchCount == 0) return arginfo;

    if(sort == MagmaDescending) {
        magmablas_sort_descend_sm_batched_kernel_driver<float>(n, dx_array, incx, dy_array, incy, dindex_array, batchCount, queue);
    }
    else {
        magmablas_sort_ascend_sm_batched_kernel_driver<float>(n, dx_array, incx, dy_array, incy, dindex_array, batchCount, queue);
    }


    return arginfo;
}


