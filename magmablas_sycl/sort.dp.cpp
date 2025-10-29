/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

*/

#include <sycl/sycl.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "swap_scalar.dp.hpp"
#include "sort.dp.hpp"

////////////////////////////////////////////////////////////////////////////////
// batch sort kernel (in shared memory with key)
// This is a trivial sort, not expected to be fast
// TODO: implement faster algorithms, like radix or bitonic sort
template<typename T>
void
sort_key_sm_kernel_batched(
    magma_sort_t sort, int n,
    T const * const *  dx_array, int incx,
    T** dy_array, int incy,
    magma_int_t** dindex_array, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local)
{
    // Do not use the commented declaration below if multiple instantiations
    // are in the same file
    //extern __shared__ T* smdata[];
    auto smdata = (unsigned char *)dpct_local;

    const int batchid = item_ct1.get_group(2);
    const int tx = item_ct1.get_local_id(2);
    const int tx2 = tx + item_ct1.get_local_range(2);

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
    item_ct1.barrier();

    sort_key_sm_device<T>(sort, n, tx, sx, sindex, item_ct1);

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
void
sort_key_kernel_batched(
    magma_sort_t sort, int n,
    T const * const *  dx_array, int incx,
    T** dy_array, int incy,
    magma_int_t** dindex_array,
    T default_value, const sycl::nd_item<3> &item_ct1, T *sB)
{

    const int batchid = item_ct1.get_group(0);
    const int bx = item_ct1.get_group(2);
    const int tx = item_ct1.get_local_id(2);
    const int gtx = bx * item_ct1.get_local_range(2) + tx;
    const int max_gtx = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);

    T* dx = (T*)dx_array[batchid];
    T* dy = dy_array[batchid];

    magma_int_t* dindex = dindex_array[batchid];

    T rA = (gtx < n) ? dx[gtx * incx] : default_value;
    T rB;
    int position = 0;

    // Loop bounds modified compared to CUDA version to ensure all threads
    // will call the barrier (otherwise, it will hang)
    for (int i = 0; i < max_gtx; i += item_ct1.get_local_range(2)) {
        int length = (i < n) ? min(item_ct1.get_local_range(2), n - i) : 0;

        if(tx < length) {
            sB[ tx ] = dx[(tx+i) * incx];
        }
        item_ct1.barrier();

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
        item_ct1.barrier();
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

        sycl::range<3> grid(1, 1, batchCount);
        sycl::range<3> threads(1, 1, nsteps);
        {

            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            sort_key_sm_kernel_batched<T>(
                                sort, n, dx_array, incx, dy_array, incy,
                                dindex_array, item_ct1,
                                dpct_local_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                        });
                });
        }
    }
    else {
        constexpr int nthreads  = 128;
        const   T default_value = (sort == MagmaDescending) ? - magmablas_get_rmax<T>() : magmablas_get_rmax<T>();
        const int blocks    = magma_ceildiv(n, nthreads);
        sycl::range<3> threads(1, 1, nthreads);

        magma_int_t max_batchCount = queue->get_maxBatch();
        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(ibatch, 1, blocks);
            {
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<T, 1> sB_acc_ct1(
                            sycl::range<1>(nthreads), cgh);

                        magma_int_t **dindex_array_i_ct6 = dindex_array + i;

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                sort_key_kernel_batched<T, nthreads>(
                                    sort, n, dx_array + i, incx, dy_array + i,
                                    incy, dindex_array_i_ct6, default_value,
                                    item_ct1,
                                    sB_acc_ct1
                                        .template get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                            });
                    });
            }
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


