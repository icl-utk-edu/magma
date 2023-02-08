/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
    // for CUDA_VERSION
#include "magma_internal.h"

#if defined(MAGMA_HAVE_HIP)
#include <hip/hip_fp16.h>
#endif

#define BLK_X 32
#define BLK_Y 4
#define MAX_BATCH    65000

dpct::global_memory<magma_int_t, 0> magma_flag(0);
dpct::global_memory<magma_int_t, 1> magma_flag_array(sycl::range<1>(MAX_BATCH),
                                                     {0});
/******************************************************************************/

void slag2h_device(
    int m, int n,
    const float *A, int lda,
    magmaHalf *HA,  int ldha,
    float rmax, magma_int_t* dinfo, sycl::nd_item<3> item_ct1)
{
    const int gtx = item_ct1.get_group(2) * BLK_X + item_ct1.get_local_id(2);
    const int gty = item_ct1.get_group(1) * BLK_Y + item_ct1.get_local_id(1);

    float tmp;
    float neg_rmax = - rmax;

    for (int j = gty; j < n; j += item_ct1.get_group_range(1) * BLK_Y) {
        for (int i = gtx; i < m; i += item_ct1.get_group_range(2) * BLK_X) {
            tmp = A[j * lda + i];
            if ( (MAGMA_S_REAL(tmp) < neg_rmax) || (MAGMA_S_REAL(tmp) > rmax) ) {
                *dinfo  = 1;
            }
            HA[j * ldha + i] =
                sycl::vec<float, 1>{tmp}
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        }
    }
}


/******************************************************************************/


void slag2h_kernel(
        int m, int n,
        float const *dA, int lda,
        magmaHalf* dHA, int ldha,
        float rmax, magma_int_t* dinfo , sycl::nd_item<3> item_ct1)
{
    slag2h_device(m, n, dA, lda, dHA, ldha, rmax, dinfo, item_ct1);
}


/******************************************************************************/


void slag2h_kernel_batched(
        int m, int n,
        float const * const * dAarray, int lda,
        magmaHalf** dHAarray, int ldha,
        float rmax, magma_int_t* dinfo_array,
        magma_queue_t queue , sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    slag2h_device(m, n, dAarray[batchid], lda, dHAarray[batchid], ldha, rmax,
                  &dinfo_array[batchid], item_ct1);
}

/******************************************************************************/
extern "C" void
magmablas_slag2h(
    magma_int_t m, magma_int_t n,
    float const * dA, magma_int_t lda,
    magmaHalf* dHA, magma_int_t ldha,
    magma_int_t *info, magma_queue_t queue)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldha < max(1,m) )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    q_ct1.memcpy(magma_flag.get_ptr(), info, sizeof(magma_flag))
        .wait(); // magma_flag = 0

    // there is no lapackf77_hlamch, please visit:
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float rmax = (float)(65504);

    sycl::range<3> threads(1, BLK_Y, BLK_X);
    sycl::range<3> grid(1, min(65000, magma_ceildiv(n, BLK_Y)),
                        magma_ceildiv(m, BLK_X));

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto magma_flag_ct7 = magma_flag.get_ptr();

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             slag2h_kernel(m, n, dA, lda, dHA, ldha, rmax,
                                           magma_flag_ct7, item_ct1);
                         });
    });

    q_ct1.memcpy(info, magma_flag.get_ptr(), sizeof(magma_flag))
        .wait(); // info = magma_flag
}


/******************************************************************************/
extern "C" void
magmablas_slag2h_batched(
    magma_int_t m, magma_int_t n,
    float const * const * dAarray, magma_int_t lda,
    magmaHalf** dHAarray, magma_int_t ldha,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    magma_int_t arginfo = 0;
    if ( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( lda < max(1,m) )
        arginfo = -4;
    else if ( ldha < max(1,m) )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    memset( info_array, 0, batchCount * sizeof(magma_int_t) );    // init info_array to zero

    // there is no lapackf77_hlamch, please visit:
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float rmax = (float)(65504);

    sycl::range<3> threads(1, BLK_Y, BLK_X);
    const int maxBatch = MAX_BATCH;
    for(int i = 0; i < batchCount; i+=maxBatch){
        magma_int_t batch = min(maxBatch, batchCount-i);
        q_ct1
            .memcpy(magma_flag_array.get_ptr(), info_array + i,
                    sizeof(magma_int_t))
            .wait();

        sycl::range<3> grid(batch, magma_ceildiv(n, BLK_Y),
                            magma_ceildiv(m, BLK_X));
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto magma_flag_array_ct7 = magma_flag_array.get_ptr();

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     slag2h_kernel_batched(
                                         m, n, dAarray + i, lda, dHAarray + i,
                                         ldha, rmax, magma_flag_array_ct7,
                                         queue, item_ct1);
                                 });
            });

        q_ct1
            .memcpy(info_array + i, magma_flag_array.get_ptr(),
                    sizeof(magma_int_t))
            .wait();
    }
}
