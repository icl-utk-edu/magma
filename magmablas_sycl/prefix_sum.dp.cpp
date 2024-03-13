/*
    -- MAGMA (version 1.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       
*/

// Parallel prefix sum (scan)
// Based on original implementation by Mark Harris, Shubhabrata Sengupta, and John D. Owens 
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

// The maximum supported input vector length is (SCAN_SEG_SIZE^2)
#define SCAN_TB_SIZE    (512)
#define SCAN_SEG_SIZE   (2*SCAN_TB_SIZE)

// ==== Kernels ==========================================================================
void prefix_sum_kernel(magma_int_t *ivec, magma_int_t *ovec, magma_int_t length, magma_int_t* workspace, magma_int_t flag,
                       sycl::nd_item<3> item_ct1, magma_int_t *sdata)
{
    const int tx = item_ct1.get_local_id(2);
    const int bx = item_ct1.get_group(2);
    const int pos = bx * SCAN_SEG_SIZE + tx;

    ivec += bx * SCAN_SEG_SIZE; 
    ovec += bx * SCAN_SEG_SIZE; 
    
    // zero shared memory
    sdata[tx] = 0;
    sdata[SCAN_TB_SIZE + tx] = 0;
    
    // load 1st part
    if(pos < length) sdata[tx] = ivec[tx]; 
    // load 2nd part
    if(pos+SCAN_TB_SIZE < length) sdata[SCAN_TB_SIZE + tx] = ivec[SCAN_TB_SIZE + tx];

    int offset = 1;
    #pragma unroll
    for (int d = SCAN_SEG_SIZE/2; d > 0; d /= 2) // upsweep
    {
        /*
        DPCT1065:193: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tx < d) {
            int ai = offset*(2*tx+1)-1;
            int bi = offset*(2*tx+2)-1;
            
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    
    if (tx == 0) {
        if(flag == 1) workspace[bx] = sdata[SCAN_SEG_SIZE - 1];    // store block increment 
        sdata[SCAN_SEG_SIZE - 1] = 0;    // clear the last element
    } 
    
    for (int d = 1; d < SCAN_SEG_SIZE; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        /*
        DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tx < d)
        {
            int ai = offset*(2*tx+1)-1;
            int bi = offset*(2*tx+2)-1;
            
            magma_int_t t   = sdata[ai];
            sdata[ai]  = sdata[bi];
            sdata[bi] += t;
        }
    }

    /*
    DPCT1065:192: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write results to device memory
    if(pos < length)              ovec[ tx ] = sdata[ tx ];
    if(pos+SCAN_TB_SIZE < length) ovec[tx+SCAN_TB_SIZE] = sdata[tx+SCAN_TB_SIZE];
}
//----------------------------------------------------------------------------------------
void prefix_update_kernel(magma_int_t *vec, magma_int_t length, magma_int_t* blk_scan_sum,
                          sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int bx = item_ct1.get_group(2);

    const int pos = (bx + 1) * SCAN_SEG_SIZE + tx; 
    magma_int_t increment = blk_scan_sum[bx + 1]; 
    
    if(pos < length)vec[pos] += increment; 
}
// ==== Internal routines ================================================================
void 
magma_prefix_sum_internal_w(
        magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, 
        magma_int_t* workspace, magma_int_t lwork, 
        magma_queue_t queue)
{
    magma_int_t lwork_min = ( (length+SCAN_SEG_SIZE-1) / SCAN_SEG_SIZE );
    if(lwork < lwork_min){
        printf("Error: not enough workspace for prefix sum\n");
        return;
    }
    const int nTB = lwork_min; 
    // 1st prefix sum
    sycl::range<3> threads_sum(1, 1, SCAN_TB_SIZE);
    sycl::range<3> grid_sum(1, 1, nTB);
    /*
    DPCT1049:195: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<magma_int_t, 1>
            sdata_acc_ct1(sycl::range<1>(SCAN_SEG_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid_sum * threads_sum, threads_sum),
                         [=](sycl::nd_item<3> item_ct1) {
                             prefix_sum_kernel(ivec, ovec, length, workspace, 1,
                                               item_ct1,
                                               sdata_acc_ct1.get_pointer());
                         });
    });

    if(nTB > 1)
    {
        // prefix sum on the workspace
        sycl::range<3> threads_sumw(1, 1, SCAN_TB_SIZE);
        sycl::range<3> grid_sumw(1, 1, 1);
        /*
        DPCT1049:196: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<magma_int_t, 1>
                    sdata_acc_ct1(sycl::range<1>(SCAN_SEG_SIZE), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_sumw * threads_sumw, threads_sumw),
                    [=](sycl::nd_item<3> item_ct1) {
                        prefix_sum_kernel(workspace, workspace, lwork,
                                          (magma_int_t *)NULL, 0, item_ct1,
                                          sdata_acc_ct1.get_pointer());
                    });
            });

        // update the sum
        sycl::range<3> threads_update(1, 1, SCAN_SEG_SIZE);
        sycl::range<3> grid_update(1, 1, nTB - 1);
        /*
        DPCT1049:197: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(
                sycl::nd_range<3>(grid_update * threads_update, threads_update),
                [=](sycl::nd_item<3> item_ct1) {
                    prefix_update_kernel(ovec, length, workspace, item_ct1);
                });
    }
}
//----------------------------------------------------------------------------------------
void 
magma_prefix_sum_internal(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_queue_t queue)
{
    magma_int_t nTB = ( (length+SCAN_SEG_SIZE-1) / SCAN_SEG_SIZE );
    
    magma_int_t* workspace; 
    const int lwork = nTB; 
    magma_imalloc(&workspace, lwork);
    
    magma_prefix_sum_internal_w(ivec, ovec, length, workspace, lwork, queue);
        
    if(workspace != (magma_int_t*)NULL)magma_free( workspace );
}
//----------------------------------------------------------------------------------------


// ===== Routines exposed ================================================================ 
extern "C"
void magma_prefix_sum_inplace(magma_int_t* ivec, magma_int_t length, magma_queue_t queue)
{
    magma_prefix_sum_internal(ivec, ivec, length, queue);
}
//----------------------------------------------------------------------------------------
extern "C"
void magma_prefix_sum_outofplace(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_queue_t queue)
{
    magma_prefix_sum_internal(ivec, ovec, length, queue);
}
//----------------------------------------------------------------------------------------
extern "C"
void magma_prefix_sum_inplace_w(magma_int_t* ivec, magma_int_t length, magma_int_t* workspace, magma_int_t lwork, magma_queue_t queue)
{
    magma_prefix_sum_internal_w(ivec, ivec, length, workspace, lwork, queue);
}
//----------------------------------------------------------------------------------------
extern "C"
void magma_prefix_sum_outofplace_w(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_int_t* workspace, magma_int_t lwork, magma_queue_t queue)
{
    magma_prefix_sum_internal_w(ivec, ovec, length, workspace, lwork, queue);
}
//----------------------------------------------------------------------------------------
