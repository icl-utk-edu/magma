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
#include "sort.cuh"

////////////////////////////////////////////////////////////////////////////////
// batch sort kernel (in shared memory with key)
// This is a trivial sort, not expected to be fast
// TODO: implement faster algorithms, like radix or bitonic sort
template<typename T>
__global__ void
sort_key_sm_kernel_batched(
    magma_sort_t sort, int n,
    T const * const *  dx_array, int incx,
    T** dy_array, int incy,
    magma_int_t** dindex_array)
{
    // Do not use the commented declaration below if multiple instantiations
    // are in the same file
    //extern __shared__ T* smdata[];
    extern __shared__ __align__(sizeof(T)) unsigned char smdata[];

    const int batchid = blockIdx.x;
    const int tx      = threadIdx.x;
    const int tx2     = tx + blockDim.x;

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

    sort_key_sm_device<T>
    (sort, n, tx, sx, sindex);

    // write output
    dy[tx*incy] = sx[tx];
    dindex[tx]  = (magma_int_t)sindex[tx];

    if(tx2 < n) {
        dy[tx2*incy] = sx[tx2];
        dindex[tx2]  = (magma_int_t)sindex[tx2];
    }
}

////////////////////////////////////////////////////////////////////////////////
// batch sort (with key) for any length
// This is a trivial sort, not expected to be fast
// TODO: implement faster algorithms, like radix or bitonic sort
template<typename T, int DIMX>
__global__ void
sort_key_kernel_batched(
    magma_sort_t sort, int n,
    T const * const *  dx_array, int incx,
    T** dy_array, int incy,
    magma_int_t** dindex_array,
    T default_value)
{
    __shared__ T sB[ DIMX ];

    const int batchid = blockIdx.z;
    const int bx      = blockIdx.x;
    const int tx      = threadIdx.x;
    const int gtx     = bx * blockDim.x + tx;

    T* dx = (T*)dx_array[batchid];
    T* dy = dy_array[batchid];

    magma_int_t* dindex = dindex_array[batchid];

    T rA = (gtx < n) ? dx[gtx * incx] : default_value;
    T rB;
    int position = 0;

    for(int i = 0; i < n; i+=blockDim.x) {
        int length = min(blockDim.x, n-i);

        if(tx < length) {
            sB[ tx ] = dx[(tx+i) * incx];
        }
        __syncthreads();

        if(sort == MagmaDescending) {
            for(int j = 0; j < length; j++) {
                rB = sB[ j ];
                position += ((rB > rA) ? 1 : 0);
                position += ((rB == rA && (i+j) < gtx) ? 1 : 0);
            }
        }
        else {
            for(int j = 0; j < length; j++) {
                rB = sB[ j ];
                position += ((rB < rA) ? 1 : 0);
                position += ((rB == rA && (i+j) > gtx) ? 1 : 0);
            }
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
// batch sort (with key) kernel driver
template<typename T>
static void
magmablas_sort_batched_kernel_driver(
    magma_sort_t sort, magma_int_t n,
    T const * const * dx_array, magma_int_t incx,
    T               **dy_array, magma_int_t incy,
    magma_int_t** dindex_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    if( n <= SORT_SM_MAX_LENGTH) {
        magma_int_t shmem = 0;
        shmem += n * sizeof(T);
        shmem += n * sizeof(int); // not magma_int_t

        magma_int_t nsteps = next_pow2( n ) >> 1;

        dim3 grid(batchCount, 1, 1);
        dim3 threads(nsteps, 1, 1);
        sort_key_sm_kernel_batched<T><<<grid, threads, shmem, queue->cuda_stream()>>>
        ( sort, n, dx_array, incx, dy_array, incy, dindex_array );
    }
    else {
        constexpr int nthreads  = 128;
        const   T default_value = (sort == MagmaDescending) ? - magmablas_get_rmax<T>() : magmablas_get_rmax<T>();
        const int blocks    = magma_ceildiv(n, nthreads);
        dim3 threads(nthreads, 1, 1);

        magma_int_t max_batchCount = queue->get_maxBatch();
        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( blocks, 1, ibatch );
            sort_key_kernel_batched<T, nthreads><<<grid, threads, 0, queue->cuda_stream()>>>
            (sort, n, dx_array+i, incx, dy_array+i, incy, dindex_array+i, default_value);
        }
    }
}

/***************************************************************************//**
    Purpose
    -------

    dsort_batched converts a batch of DOUBLE arrays into a corresponding batch
    of ordered sequences, and stores the corresponding key values
      ** This is an out-of-place sort
      ** This routine is originally intended for sorting the singular values/eigenvalues
         in MAGMA, with the key used for re-ordering singular/eigenvectors.
      ** Since the overhead of sorting is usually negligible in these solvers,
         this routine is not intended for high performance

    Arguments
    ----------
    @param[in]
    sort    magma_sort_t
            Sorting direction (MagmaAscending or MagmaDescending)

    @param[in]
    n       INTEGER
            The length of each array in the batch, n >= 0.

    @param[in]
    dx_array Array of pointers, dimension(batchCount)
             Each is a DOUBLE array of length n.
             On entry, the array to be sorted.
             Unchanged on exit.

    @param[in]
    incx    INTEGER
            The stride between two consecutive elements in each input array.
            INCX > 0.

    @param[out]
    dy_array Array of pointers, dimension(batchCount)
             Each is a DOUBLE array of length n.
             On exit, the sorted arrays.

    @param[in]
    incy    INTEGER
            The stride between two consecutive elements in each output array.
            INCY > 0.


    @param[out]
    dindex_array    Array of pointers, dimension(batchCount)
                    Each is an INTEGER array of length n.
                    On exit, the key values of each sorted array.

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

    // Quick return if possible
    if(n < 2 || batchCount == 0) return arginfo;

    magmablas_sort_batched_kernel_driver<double>(sort, n, dx_array, incx, dy_array, incy, dindex_array, batchCount, queue);

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------

    ssort_batched converts a batch of FLOAT arrays into a corresponding batch
    of ordered sequences, and stores the corresponding key values
      ** This is an out-of-place sort
      ** This routine is originally intended for sorting the singular values/eigenvalues
         in MAGMA, with the key used for re-ordering singular/eigenvectors.
      ** Since the overhead of sorting is usually negligible in these solvers,
         this routine is not intended for high performance

    Arguments
    ----------
    @param[in]
    sort    magma_sort_t
            Sorting direction (MagmaAscending or MagmaDescending)

    @param[in]
    n       INTEGER
            The length of each array in the batch, n >= 0.

    @param[in]
    dx_array Array of pointers, dimension(batchCount)
             Each is a FLOAT array of length n.
             On entry, the array to be sorted.
             Unchanged on exit.

    @param[in]
    incx    INTEGER
            The stride between two consecutive elements in each input array.
            INCX > 0.

    @param[out]
    dy_array Array of pointers, dimension(batchCount)
             Each is a FLOAT array of length n.
             On exit, the sorted arrays.

    @param[in]
    incy    INTEGER
            The stride between two consecutive elements in each output array.
            INCY > 0.


    @param[out]
    dindex_array    Array of pointers, dimension(batchCount)
                    Each is an INTEGER array of length n.
                    On exit, the key values of each sorted array.

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
    magma_sort_t sort, magma_int_t n,
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

    // Quick return if possible
    if(n < 2 || batchCount == 0) return arginfo;

    magmablas_sort_batched_kernel_driver<float>(sort, n, dx_array, incx, dy_array, incy, dindex_array, batchCount, queue);

    return arginfo;
}


