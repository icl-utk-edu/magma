/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define BLK_X    (128)

// =============================================================================
// Auxiliary functions for vbatched routines

/******************************************************************************/
// lu setup
// for a batch of different size matrices, extracts important info such as:
// max_m, max_n, max_min_mn, max_mxn (biggest matrix)

void magma_getrf_vbatched_setup_kernel( magma_int_t *m, magma_int_t *n, magma_int_t *stats, int batchCount ,
                                        sycl::nd_item<3> item_ct1,
                                        uint8_t *dpct_local)
{
    auto sdata = (int *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    int im = 0, in = 0, max_m = 0, max_n = 0, max_min_mn = 0, max_mxn = 0;

    // shared ptr's
    int* smax_m      = (int*)sdata;
    int* smax_n      = smax_m + ntx;
    int* smax_min_mn = smax_n + ntx;
    int* smax_mxn    = smax_min_mn + ntx;

    for(int i = tx; i < batchCount; i+=ntx) {
        im = (int)m[i];
        in = (int)n[i];
        max_m      = max(max_m, im);
        max_n      = max(max_n, in);
        max_min_mn = max(max_min_mn, min(im, in));
        max_mxn    = max(max_mxn, im*in);
    }

    smax_m[tx]      = max_m;
    smax_n[tx]      = max_n;
    smax_min_mn[tx] = max_min_mn;
    smax_mxn[tx]    = max_mxn;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // max reduce
    #pragma unroll
    for(int i = 1024; i > 0; i >>= 1) {
        if(ntx > i) {
            if ( tx < i && tx + i < ntx ) {
                smax_m[tx]      = max( smax_m[tx], smax_m[tx+i] );
                smax_n[tx]      = max( smax_n[tx], smax_n[tx+i] );
                smax_min_mn[tx] = max( smax_min_mn[tx], smax_min_mn[tx+i] );
                smax_mxn[tx]    = max( smax_mxn[tx], smax_mxn[tx+i] );
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    stats[0] = (magma_int_t)smax_m[0];
    stats[1] = (magma_int_t)smax_n[0];
    stats[2] = (magma_int_t)smax_min_mn[0];
    stats[3] = (magma_int_t)smax_mxn[0];
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_getrf_vbatched_setup(
            magma_int_t *m, magma_int_t *n, magma_int_t *stats,
            magma_int_t batchCount, magma_queue_t queue )
{
    const int nthreads =  min(batchCount, 512);
    /*
    DPCT1083:202: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int shmem = nthreads * 4 * sizeof(int);
    /*
    DPCT1049:201: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nthreads),
                                           sycl::range<3>(1, 1, nthreads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_getrf_vbatched_setup_kernel(
                                 m, n, stats, batchCount, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
// max reduce kernel
// 1) set overwrite to 0 ==> result is written to y and x is untouched
//    set overwrite to 1 ==> result is written to x (x is destroyed)
// Each thread block gets the max of <MAX_REDUCE_SEGMENT> elements and
// writes it to the workspace
#define MAX_REDUCE_SEGMENT    (512)    // must be even
#define MAX_REDUCE_TX         (MAX_REDUCE_SEGMENT/2)


void magma_ivec_max_kernel( int vecsize,
                              magma_int_t* x, magma_int_t* y,
                              int overwrite, sycl::nd_item<3> item_ct1,
                              int *swork)
{
    const int tx = item_ct1.get_local_id(2);
    const int bx = item_ct1.get_group(2);
    const int gtx = bx * MAX_REDUCE_SEGMENT + tx;

    // init shmem
    swork[tx] = 0;
    swork[tx + MAX_REDUCE_TX] = 0;

    // read the input segment into swork
    if(gtx < vecsize)swork[tx] = (int)x[gtx];
    if( (gtx + MAX_REDUCE_TX) < vecsize ) swork[tx + MAX_REDUCE_TX] = (int)x[gtx + MAX_REDUCE_TX];
    /*
    DPCT1065:203: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    magma_max_reduce<MAX_REDUCE_SEGMENT, int>(tx, swork, item_ct1);
    /*
    DPCT1065:204: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // write the result back
    if(overwrite == 0)
    {
        if(tx == 0) y[bx] = (magma_int_t)swork[0];
    }
    else
    {
        if(tx == 0) x[bx] = (magma_int_t)swork[0];
    }
}

//----------------
// kernel driver
//----------------
extern "C"
magma_int_t magma_ivec_max( magma_int_t vecsize,
                              magma_int_t* x,
                              magma_int_t* work, magma_int_t lwork, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, MAX_REDUCE_TX);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, MAX_REDUCE_SEGMENT));
    if (lwork < (magma_int_t)grid[2]) {
        printf("error in %s: lwork must be at least %lld, input is %lld\n",
               __func__, (long long)grid[2], (long long)lwork);
    }

    /*
    DPCT1049:205: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            swork_acc_ct1(sycl::range<1>(MAX_REDUCE_SEGMENT), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_ivec_max_kernel(vecsize, x, work, 0,
                                                   item_ct1,
                                                   swork_acc_ct1.get_pointer());
                         });
    });
    magma_int_t new_vecsize = grid[2];

    while(new_vecsize > 1)
    {
        grid[2] = magma_ceildiv(new_vecsize, MAX_REDUCE_SEGMENT);
        /*
        DPCT1049:206: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    swork_acc_ct1(sycl::range<1>(MAX_REDUCE_SEGMENT),
                                  cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_ivec_max_kernel(
                                         new_vecsize, work, (magma_int_t *)NULL,
                                         1, item_ct1,
                                         swork_acc_ct1.get_pointer());
                                 });
            });
        new_vecsize = grid[2];
    }

    // copy the result to cpu and return it
    magma_int_t vecmax = 0;
    magma_getvector(1, sizeof(magma_int_t), work, 1, &vecmax, 1, queue);
    return (magma_int_t)vecmax;
}


/******************************************************************************/
// integer sum (isum) reduce kernel
// initially needed for vbatched trsm
// 1) set overwrite to 0 ==> result is written to y and x is untouched
//    set overwrite to 1 ==> result is written to x (x is destroyed)
// Each thread block gets the custom sum of <ISUM_REDUCE_SEGMENT> elements and
// writes it to the workspace
#define ISUM_REDUCE_SEGMENT    (512)    // must be even
#define ISUM_REDUCE_TX         (ISUM_REDUCE_SEGMENT/2)


void magma_isum_reduce_kernel( int vecsize,
                              magma_int_t* x, magma_int_t* y,
                              int overwrite, sycl::nd_item<3> item_ct1,
                              int *swork)
{
    const int tx = item_ct1.get_local_id(2);
    const int bx = item_ct1.get_group(2);
    const int gtx = bx * ISUM_REDUCE_SEGMENT + tx;

    // init shmem
    swork[tx] = 0;
    swork[tx + ISUM_REDUCE_TX] = 0;

    // read the input segment into swork
    if(gtx < vecsize)swork[tx] = (int)(x[gtx]);
    if( (gtx + ISUM_REDUCE_TX) < vecsize ) swork[tx + ISUM_REDUCE_TX] = (int)(x[gtx + ISUM_REDUCE_TX]);
    /*
    DPCT1065:207: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    magma_sum_reduce<ISUM_REDUCE_SEGMENT, int>(tx, swork, item_ct1);
    /*
    DPCT1065:208: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // write the result back
    if(overwrite == 0)
    {
        if(tx == 0) y[bx] = (magma_int_t)swork[0];
    }
    else
    {
        if(tx == 0) x[bx] = (magma_int_t)swork[0];
    }
}

//----------------
// kernel driver
//----------------
extern "C"
magma_int_t magma_isum_reduce( magma_int_t vecsize,
                              magma_int_t* x,
                              magma_int_t* work, magma_int_t lwork, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, ISUM_REDUCE_TX);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, ISUM_REDUCE_SEGMENT));
    if (lwork < (magma_int_t)grid[2]) {
        printf("error in %s: lwork must be at least %lld, input is %lld\n",
               __func__, (long long)grid[2], (long long)lwork);
    }

    /*
    DPCT1049:209: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            swork_acc_ct1(sycl::range<1>(ISUM_REDUCE_SEGMENT), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_isum_reduce_kernel(
                                 vecsize, x, work, 0, item_ct1,
                                 swork_acc_ct1.get_pointer());
                         });
    });
    magma_int_t new_vecsize = grid[2];

    while(new_vecsize > 1)
    {
        grid[2] = magma_ceildiv(new_vecsize, ISUM_REDUCE_SEGMENT);
        /*
        DPCT1049:210: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    swork_acc_ct1(sycl::range<1>(ISUM_REDUCE_SEGMENT),
                                  cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_isum_reduce_kernel(
                                         new_vecsize, work, (magma_int_t *)NULL,
                                         1, item_ct1,
                                         swork_acc_ct1.get_pointer());
                                 });
            });
        new_vecsize = grid[2];
    }

    // copy the result to cpu and return it
    magma_int_t isum = 0;
    magma_getvector(1, sizeof(magma_int_t), work, 1, &isum, 1, queue);
    return (magma_int_t)isum;
}


/******************************************************************************/
// y[i] = a1 * x1[i] + a2 * x2[i]

void magma_ivec_add_kernel( int vecsize,
                                  magma_int_t a1, magma_int_t *x1,
                                  magma_int_t a2, magma_int_t *x2,
                                  magma_int_t *y, sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        y[indx] = a1 * x1[indx] + a2 * x2[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_add( magma_int_t vecsize,
                           magma_int_t a1, magma_int_t *x1,
                           magma_int_t a2, magma_int_t *x2,
                           magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));
    /*
    DPCT1049:211: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_add_kernel(vecsize, a1, x1, a2, x2, y,
                                                 item_ct1);
                       });
}


/******************************************************************************/
// y[i] = x1[i] * x2[i]

void magma_ivec_mul_kernel( int vecsize,
                                  magma_int_t *x1,
                                  magma_int_t *x2,
                                  magma_int_t *y,
                                  sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        y[indx] = x1[indx] * x2[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_mul( magma_int_t vecsize,
                           magma_int_t *x1, magma_int_t *x2,
                           magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));
    /*
    DPCT1049:212: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_mul_kernel(vecsize, x1, x2, y, item_ct1);
                       });
}


/******************************************************************************/
// ceildiv
void magma_ivec_ceildiv_kernel(int vecsize, magma_int_t *x, int nb, magma_int_t *y,
                               sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        y[indx] = (magma_int_t)magma_ceildiv(x[indx], (magma_int_t)nb);
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_ceildiv( magma_int_t vecsize,
                        magma_int_t *x,
                        magma_int_t nb,
                        magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:213: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_ceildiv_kernel(vecsize, x, nb, y,
                                                     item_ct1);
                       });
}


/******************************************************************************/
// roundup

void magma_ivec_roundup_kernel(int vecsize, magma_int_t *x, int nb, magma_int_t *y,
                               sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        y[indx] = (magma_int_t)magma_roundup(x[indx], (magma_int_t)nb);
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_roundup( magma_int_t vecsize,
                        magma_int_t *x,
                        magma_int_t nb,
                        magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:214: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_roundup_kernel(vecsize, x, nb, y,
                                                     item_ct1);
                       });
}


/******************************************************************************/
// set vector to a const value
template<typename T>

void magma_setvector_const_gpu_kernel(int vecsize, T *x, T value,
                                      sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        x[indx] = value;
    }
}

//----------------
// kernel drivers
//----------------
extern "C"
void magma_ivec_setc( magma_int_t vecsize,
                                magma_int_t *x,
                                magma_int_t value,
                                magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:215: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_setvector_const_gpu_kernel<magma_int_t>(
                               vecsize, x, value, item_ct1);
                       });
}

//---------------
extern "C"
void magma_zsetvector_const( magma_int_t vecsize,
                                magmaDoubleComplex *x,
                                magmaDoubleComplex value,
                                magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:216: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_setvector_const_gpu_kernel<magmaDoubleComplex>(
                               vecsize, x, value, item_ct1);
                       });
}

//---------------
extern "C"
void magma_csetvector_const( magma_int_t vecsize,
                                magmaFloatComplex *x,
                                magmaFloatComplex value,
                                magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:217: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_setvector_const_gpu_kernel<magmaFloatComplex>(
                               vecsize, x, value, item_ct1);
                       });
}

//---------------
extern "C"
void magma_dsetvector_const( magma_int_t vecsize,
                                double *x,
                                double value,
                                magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:218: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_setvector_const_gpu_kernel<double>(
                               vecsize, x, value, item_ct1);
                       });
}

//---------------
extern "C"
void magma_ssetvector_const( magma_int_t vecsize,
                                float *x,
                                float value,
                                magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:219: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_setvector_const_gpu_kernel<float>(
                               vecsize, x, value, item_ct1);
                       });
}


/******************************************************************************/
// performs addition with a const value

void magma_ivec_addc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y,
                            sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        y[indx] = (x[indx] + (magma_int_t)value);
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_addc(magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:220: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_addc_kernel(vecsize, x, value, y,
                                                  item_ct1);
                       });
}


/******************************************************************************/
// performs multiplication with a const value

void magma_ivec_mulc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y,
                            sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize)
    {
        y[indx] = (x[indx] * (magma_int_t)value);
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_mulc(magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:221: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_mulc_kernel(vecsize, x, value, y,
                                                  item_ct1);
                       });
}


/******************************************************************************/
// performs a min. operation against a const value

void magma_ivec_minc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y,
                            sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const magma_int_t value_l = (magma_int_t)value;
    if(indx < vecsize)
    {
        y[indx] = ( value_l < x[indx] )? value_l : x[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_minc(magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:222: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_minc_kernel(vecsize, x, value, y,
                                                  item_ct1);
                       });
}


/******************************************************************************/
// performs an element-wise min. operation between two vectors
// result is stored in another vector (could be either of the two input vectors)

void magma_ivec_min_vv_kernel(int vecsize, magma_int_t *v1, magma_int_t *v2, magma_int_t *y,
                              sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if(indx < vecsize) {
        y[indx] = min( v1[indx], v2[indx] );
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_min_vv(magma_int_t vecsize, magma_int_t *v1, magma_int_t *v2, magma_int_t *y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:223: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_min_vv_kernel(vecsize, v1, v2, y,
                                                    item_ct1);
                       });
}

/******************************************************************************/
// performs a max. operation against a const value

void magma_ivec_maxc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y,
                            sycl::nd_item<3> item_ct1)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const magma_int_t value_l = (magma_int_t)value;
    if(indx < vecsize)
    {
        y[indx] = ( value_l > x[indx] )? value_l : x[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C"
void magma_ivec_maxc(magma_int_t vecsize, magma_int_t* x, magma_int_t value, magma_int_t* y, magma_queue_t queue)
{
    sycl::range<3> threads(1, 1, BLK_X);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, BLK_X));

    /*
    DPCT1049:224: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_ivec_maxc_kernel(vecsize, x, value, y,
                                                  item_ct1);
                       });
}


/******************************************************************************/
// This kernel is for the vbatched trsm routine
// auxiliary kernel to compute jb = (m % TRI_BATCHED_NB == 0) ? TRI_BATCHED_NB : (m % TRI_BATCHED_NB)
// This kernel is specific to trsm, so it is not in vbatched_aux.h
void magma_compute_trsm_jb_kernel(int vecsize, magma_int_t *m, int tri_nb, magma_int_t *jbv,
                                  sycl::nd_item<3> item_ct1)
//(int vecsize, int* m, int tri_nb, int* jbv)
{
    const int indx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const int my_m = (magma_int_t)m[indx];
    if(indx < vecsize)
    {
        int my_jb;
        if(my_m % tri_nb == 0) my_jb = tri_nb;
        else my_jb = (my_m % tri_nb);

        jbv[indx] = (magma_int_t)my_jb;
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_compute_trsm_jb(magma_int_t vecsize, magma_int_t* m, magma_int_t tri_nb, magma_int_t* jbv, magma_queue_t queue)
{
    const int nthreads = 128;
    sycl::range<3> threads(1, 1, nthreads);
    sycl::range<3> grid(1, 1, magma_ceildiv(vecsize, nthreads));

    /*
    DPCT1049:225: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_compute_trsm_jb_kernel(vecsize, m, tri_nb, jbv,
                                                        item_ct1);
                       });
}


/******************************************************************************/
// A max-reduce kernel specific for computing the max M/N/K for launching vbatched kernels
#define AUX_MAX_SEGMENT    (256)    // must be even
#define AUX_MAX_TX         (AUX_MAX_SEGMENT)
void magma_imax_size_kernel_1(magma_int_t *n, int l, sycl::nd_item<3> item_ct1,
                              int *swork)
{
    magma_int_t *vec;
    const int tx = item_ct1.get_local_id(2);
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;

    vec = n;
    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    swork[tx] = lmax;
    /*
    DPCT1065:226: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    magma_max_reduce<AUX_MAX_SEGMENT, int>(tx, swork, item_ct1);
    // no need to sync
    if(tx == 0){
        vec[l] = (magma_int_t)(swork[0]);
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_imax_size_1(magma_int_t *n, magma_int_t l, magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, 1);
    sycl::range<3> threads(1, 1, AUX_MAX_TX);
    /*
    DPCT1049:227: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            swork_acc_ct1(sycl::range<1>(AUX_MAX_SEGMENT), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_imax_size_kernel_1(
                                 n, l, item_ct1, swork_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
void magma_imax_size_kernel_2(magma_int_t *m, magma_int_t *n, int l,
                              sycl::nd_item<3> item_ct1, int *swork)
{
    magma_int_t *vec;
    const int bx = item_ct1.get_group(2);
    const int tx = item_ct1.get_local_id(2);
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;

    if     (bx == 0) vec = m;
    else if(bx == 1) vec = n;

    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    swork[tx] = lmax;
    /*
    DPCT1065:228: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    magma_max_reduce<AUX_MAX_SEGMENT, int>(tx, swork, item_ct1);
    // no need to sync
    if(tx == 0){
        vec[l] = (magma_int_t)(swork[0]);
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_imax_size_2(magma_int_t *m, magma_int_t *n, magma_int_t l, magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, 2);
    sycl::range<3> threads(1, 1, AUX_MAX_TX);
    /*
    DPCT1049:229: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            swork_acc_ct1(sycl::range<1>(AUX_MAX_SEGMENT), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_imax_size_kernel_2(
                                 m, n, l, item_ct1,
                                 swork_acc_ct1.get_pointer());
                         });
    });
}


/******************************************************************************/
void magma_imax_size_kernel_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, int l,
                              sycl::nd_item<3> item_ct1, int *swork)
{
    magma_int_t *vec;
    const int bx = item_ct1.get_group(2);
    const int tx = item_ct1.get_local_id(2);
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;

    if     (bx == 0) vec = m;
    else if(bx == 1) vec = n;
    else if(bx == 2) vec = k;

    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    swork[tx] = lmax;
    /*
    DPCT1065:230: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    magma_max_reduce<AUX_MAX_SEGMENT, int>(tx, swork, item_ct1);
    // no need to sync
    if(tx == 0){
        vec[l] = (magma_int_t)(swork[0]);
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_imax_size_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, magma_int_t l, magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, 3);
    sycl::range<3> threads(1, 1, AUX_MAX_TX);
    /*
    DPCT1049:231: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            swork_acc_ct1(sycl::range<1>(AUX_MAX_SEGMENT), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_imax_size_kernel_3(
                                 m, n, k, l, item_ct1,
                                 swork_acc_ct1.get_pointer());
                         });
    });
}
