/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Weifeng Liu

*/

// CSR5 SpMV kernel
// see paper by W. Liu and B. Vinter. (2015).
// "CSR5: An Efficient Storage Format for Cross-Platform 
//  Sparse Matrix-Vector Multiplication". 
// 29th ACM International Conference on Supercomputing (ICS15). pp. 339-350.

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"
#include "atomicopsmagmaDoubleComplex.h"
#include <cmath>


#define MAGMA_CSR5_THREAD_GROUP 128
#define MAGMA_CSR5_THREAD_BUNCH 32

__inline__ void
sum_32(
             magmaDoubleComplex *s_sum,
    const    int                 local_id)
{
    if (local_id < 16)   s_sum[local_id] += s_sum[local_id + 16];
    if (local_id < 8)    s_sum[local_id] += s_sum[local_id + 8];
    if (local_id < 4)    s_sum[local_id] += s_sum[local_id + 4];
    if (local_id < 2)    s_sum[local_id] += s_sum[local_id + 2];
    if (local_id < 1)    s_sum[local_id] += s_sum[local_id + 1];
}

__inline__ void
scan_32(
             magmaDoubleComplex *s_scan,
    const    int                 local_id)
{
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    magmaDoubleComplex temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     
                          s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   
                          s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   
                          s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   
                          s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) {
        s_scan[31] = s_scan[15]; s_scan[15] = sycl::double2(r, i);
    }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   
                          temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; 
                          s_scan[bi] += temp; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   
                          temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; 
                          s_scan[bi] += temp; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   
                          temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; 
                          s_scan[bi] += temp; }
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   
                          temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; 
                          s_scan[bi] += temp; }
}

__inline__ magmaDoubleComplex
candidate(
          magmaDoubleComplex      *d_value_tile,
          magmaDoubleComplex      *d_x,
    const magma_index_t           *d_column_index_tile,
    const magma_index_t            candidate_index,
    const magmaDoubleComplex       alpha)
{
    magmaDoubleComplex x = sycl::double2(r, i);
    x = d_x[d_column_index_tile[candidate_index]];
    return d_value_tile[candidate_index] * x * alpha;
}


__dpct_inline__ magmaDoubleComplex
segmented_sum(magmaDoubleComplex tmp_sum, magmaDoubleComplex *s_sum,
              const magma_index_t scansum_offset, const magma_index_t lane_id)
{
    if (lane_id)
        s_sum[lane_id - 1] = tmp_sum;
    s_sum[lane_id] =
        lane_id == MAGMA_CSR5_OMEGA - 1 ? sycl::double2(r, i) : s_sum[lane_id];
    magmaDoubleComplex sum = tmp_sum = s_sum[lane_id];
    scan_32(s_sum, lane_id); // exclusive scan
    s_sum[lane_id] += tmp_sum; // inclusive scan (exclusive scan+original val)
    tmp_sum = s_sum[lane_id + scansum_offset];
    tmp_sum = tmp_sum - s_sum[lane_id] + sum;

    return tmp_sum;
}

template<int c_sigma>
__inline__ void 
tile_fast_track(
          magmaDoubleComplex    *d_value_tile,
          magmaDoubleComplex    *d_x,
    const magma_index_t         *d_column_index_tile,
          magmaDoubleComplex    *d_calibrator,
          magmaDoubleComplex    *s_sum,
    const int                    lane_id,
    const magma_index_t          par_id,
    const magmaDoubleComplex     alpha)
{
    magmaDoubleComplex sum = sycl::double2(r, i);

#pragma unroll
    for (int i = 0; i < c_sigma; i++)
    {
        sum += candidate(d_value_tile, d_x, d_column_index_tile, 
                         i * MAGMA_CSR5_OMEGA + lane_id, alpha);
    }

    s_sum[lane_id] = sum;
    sum_32(s_sum, lane_id);
    if (!lane_id)
    {
        d_calibrator[par_id] = s_sum[0];
    }
}

template<int c_sigma>
__inline__ void 
tile_normal_track(
    const magma_index_t           *d_column_index_tile,
          magmaDoubleComplex      *d_value_tile,
          magmaDoubleComplex      *d_x,
    const magma_uindex_t          *d_tile_desc,
    const magma_index_t           *d_tile_desc_offset_ptr,
    const magma_index_t           *d_tile_desc_offset,
          magmaDoubleComplex      *d_calibrator,
          magmaDoubleComplex      *d_y,
          magmaDoubleComplex      *s_sum,
    volatile int                  *s_scan,
    const magma_index_t            par_id,
    const int                      lane_id,
    const int                      bit_y_offset,
    const int                      bit_scansum_offset,
    const bool                     empty_rows,
    const magmaDoubleComplex       alpha)
{
    int start = 0;
    int stop = 0;

    bool local_bit;
    magmaDoubleComplex sum = sycl::double2(r, i);

    magma_index_t offset_pointer = empty_rows ? 
                                   d_tile_desc_offset_ptr[par_id] : 0;

    magma_uindex_t descriptor = d_tile_desc[lane_id];

    magma_index_t y_offset = descriptor >> (32 - bit_y_offset);
    const int scansum_offset = (descriptor << bit_y_offset) 
                               >> (32 - bit_scansum_offset);
    const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;

    bool direct = false;

    magmaDoubleComplex first_sum, last_sum;

    // step 1. thread-level seg sum

    int ly = 0;

    // extract the first bit-flag packet
    descriptor = descriptor << (bit_y_offset + bit_scansum_offset);
    descriptor = lane_id ? descriptor : descriptor | 0x80000000;

    local_bit = (descriptor >> 31) & 0x1;
    start = !local_bit;
    direct = local_bit & (bool)lane_id;

    sum = candidate(d_value_tile, d_x, 
                    d_column_index_tile, lane_id, alpha);

    #pragma unroll
    for (int i = 1; i < c_sigma; i++)
    {
        int norm_i = i - bit_bitflag;

        if (!(ly || norm_i) || (ly && !(31 & norm_i)))
        {
            ly++;
            descriptor = d_tile_desc[ly * MAGMA_CSR5_OMEGA + lane_id];
        }
        norm_i = !ly ? 31 & i : 31 & norm_i;
        norm_i = 31 - norm_i;

        local_bit = (descriptor >> norm_i) & 0x1;

        if (local_bit)
        {
            if (direct)
                d_y[empty_rows ? d_tile_desc_offset[offset_pointer + y_offset] 
                                 : y_offset] += sum;
            else
                first_sum = sum;
        }

        y_offset += local_bit & direct;

        direct |= local_bit;
        sum = local_bit ? sycl::double2(r, i) : sum;
        stop += local_bit;

        sum += candidate(d_value_tile, d_x, d_column_index_tile, 
                         i * MAGMA_CSR5_OMEGA + lane_id, alpha);
    }

    first_sum = direct ? first_sum : sum;
    last_sum = sum;

    // step 2. segmented sum
    sum = start ? first_sum : sycl::double2(r, i);

    sum = segmented_sum(sum, s_sum, scansum_offset, lane_id);

    // step 3-1. add s_sum to position stop
    last_sum += (start <= stop) ? sum : sycl::double2(r, i);

    // step 3-2. write sums to result array
    if (direct)
        d_y[empty_rows ? d_tile_desc_offset[offset_pointer + y_offset] 
                       : y_offset] += last_sum;

    // the first/last value of the first thread goes to calibration
    if (!lane_id)
        d_calibrator[par_id] = direct ? first_sum : last_sum;
}

template<int c_sigma>
__inline__ void 
spmv_tile(
    const magma_index_t           *d_column_index_tile,
          magmaDoubleComplex      *d_value_tile,
    const magma_index_t           *d_row_pointer,
          magmaDoubleComplex      *d_x,
    const magma_uindex_t          *d_tile_ptr,
    const magma_uindex_t          *d_tile_desc,
    const magma_index_t           *d_tile_desc_offset_ptr,
    const magma_index_t           *d_tile_desc_offset,
          magmaDoubleComplex      *d_calibrator,
          magmaDoubleComplex      *d_y,
    const magma_index_t            par_id,
    const int                      lane_id,
    const int                      bunch_id,
    const int                      bit_y_offset,
    const int                      bit_scansum_offset,
    const magmaDoubleComplex       alpha,
    const sycl::nd_item<3> &item_ct1,
    magmaDoubleComplex *s_sum,
    volatile int *s_scan,
    volatile magma_uindex_t *s_row_start_stop)
{
    magma_uindex_t row_start, row_stop;

    if (item_ct1.get_local_id(2) <
        MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1)
    {
        s_row_start_stop[item_ct1.get_local_id(2)] =
            d_tile_ptr[par_id + item_ct1.get_local_id(2)];
    }
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    row_start = s_row_start_stop[bunch_id];
    row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;

    if (row_start == row_stop) // fast track through reduction
    {
        tile_fast_track<c_sigma>
                (d_value_tile, d_x, d_column_index_tile, d_calibrator,
                 &s_sum[bunch_id * MAGMA_CSR5_OMEGA],
                 lane_id, par_id, alpha);
    }
    else
    {
        const bool empty_rows = (row_start >> 31) & 0x1;
        row_start &= 0x7FFFFFFF;

        d_y = &d_y[row_start+1];

        tile_normal_track<c_sigma>
                (d_column_index_tile, d_value_tile, d_x,
                 d_tile_desc, d_tile_desc_offset_ptr,
                 d_tile_desc_offset, d_calibrator, d_y,
                 &s_sum[bunch_id * MAGMA_CSR5_OMEGA],
                 &s_scan[bunch_id * (MAGMA_CSR5_OMEGA + 1)],
                 par_id, lane_id,
                 bit_y_offset, bit_scansum_offset, empty_rows, alpha);
    }
}

template<int c_sigma>
void 
spmv_csr5_compute_kernel(
    const magma_index_t           *d_column_index,
          magmaDoubleComplex      *d_value,
    const magma_index_t           *d_row_pointer,
          magmaDoubleComplex      *d_x,
    const magma_uindex_t          *d_tile_ptr,
    const magma_uindex_t          *d_tile_desc,
    const magma_index_t           *d_tile_desc_offset_ptr,
    const magma_index_t           *d_tile_desc_offset,
          magmaDoubleComplex      *d_calibrator,
          magmaDoubleComplex      *d_y,
    const magma_index_t            p,
    const int                      num_packet,
    const int                      bit_y_offset,
    const int                      bit_scansum_offset,
    const magmaDoubleComplex       alpha,
    const sycl::nd_item<3> &item_ct1,
    magmaDoubleComplex *s_sum,
    int *s_scan,
    magma_uindex_t *s_row_start_stop)
{
    // warp lane id
    const int lane_id =
        31 & item_ct1.get_local_id(2); // threadIdx.x % CSR5_OMEGA;
    // warp global id == par_id
    const magma_index_t par_id =
        (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
         item_ct1.get_local_id(2)) /
        MAGMA_CSR5_OMEGA;
    const int bunch_id = item_ct1.get_local_id(2) / MAGMA_CSR5_OMEGA;

    if (par_id >= p - 1)
        return;

    spmv_tile<c_sigma>(
        &d_column_index[par_id * MAGMA_CSR5_OMEGA * c_sigma],
        &d_value[par_id * MAGMA_CSR5_OMEGA * c_sigma], d_row_pointer, d_x,
        d_tile_ptr, &d_tile_desc[par_id * MAGMA_CSR5_OMEGA * num_packet],
        d_tile_desc_offset_ptr, d_tile_desc_offset, d_calibrator, d_y, par_id,
        lane_id, bunch_id, bit_y_offset, bit_scansum_offset, alpha, item_ct1,
        s_sum, s_scan, s_row_start_stop);
}

void 
spmv_csr5_calibrate_kernel(
    const magma_uindex_t      *d_tile_ptr,
    const magmaDoubleComplex  *d_calibrator,
          magmaDoubleComplex  *d_y,
    const magma_index_t        p,
    const sycl::nd_item<3> &item_ct1,
    volatile magma_index_t *s_tile_ptr,
    magmaDoubleComplex *s_calibrator)
{
    //const int lane_id  = threadIdx.x % MAGMA_CSR5_THREAD_BUNCH;
    //const int bunch_id = threadIdx.x / MAGMA_CSR5_THREAD_BUNCH;
    const int local_id = item_ct1.get_local_id(2);
    const magma_index_t global_id =
        item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);

    magmaDoubleComplex sum;

    //volatile __shared__ 
    //         magmaDoubleComplex  s_sum[MAGMA_CSR5_THREAD_GROUP 
    //                                   / MAGMA_CSR5_THREAD_BUNCH];

    s_tile_ptr[local_id] = global_id < p-1 ? 
                  (magma_index_t)(d_tile_ptr[global_id] & 0x7FFFFFFF) : -1;
    s_calibrator[local_id] = sum =
        global_id < p - 1 ? d_calibrator[global_id] : sycl::double2(r, i);
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // do a fast track if all s_tile_ptr are the same
    if (s_tile_ptr[0] == s_tile_ptr[MAGMA_CSR5_THREAD_GROUP - 1])
    {
        //sum = sum_32_shfl<vT>(sum);
        //if (!lane_id)
        //    s_sum[bunch_id] = sum;
        //__syncthreads();

        //if (!bunch_id)
        //{
        //    sum = lane_id < (MAGMA_CSR5_THREAD_GROUP 
        //                     / MAGMA_CSR5_THREAD_BUNCH) ? s_sum[lane_id] : 0;
        //    sum = sum_32_shfl<vT>(sum);
        //}

        if (local_id < 64) s_calibrator[local_id] += s_calibrator[local_id+64];
        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (local_id < 32) s_calibrator[local_id] += s_calibrator[local_id+32];
        if (local_id < 16) s_calibrator[local_id] += s_calibrator[local_id+16];
        if (local_id < 8) s_calibrator[local_id] += s_calibrator[local_id+8];
        if (local_id < 4) s_calibrator[local_id] += s_calibrator[local_id+4];
        if (local_id < 2) s_calibrator[local_id] += s_calibrator[local_id+2];
        if (local_id < 1) s_calibrator[local_id] += s_calibrator[local_id+1];

        if (!local_id)
        {
            atomicAddmagmaDoubleComplex(&d_y[s_tile_ptr[0]], s_calibrator[0]);
        }
        return;
    }

    int local_par_id = local_id;
    magma_index_t row_start_current, row_start_target, row_start_previous;
    sum = sycl::double2(r, i);

    // use (p - 1), due to the tail tile is dealt with CSR-vector method
    if (global_id < p - 1)
    {
        row_start_previous = local_id ? s_tile_ptr[local_id-1] : -1;
        row_start_current = s_tile_ptr[local_id];

        if (row_start_previous != row_start_current)
        {
            row_start_target = row_start_current;

            while (row_start_target == row_start_current &&
                   local_par_id < item_ct1.get_local_range(2))
            {
                sum +=  s_calibrator[local_par_id];
                local_par_id++;
                row_start_current = s_tile_ptr[local_par_id];
            }
            if (row_start_target == s_tile_ptr[0] 
                || row_start_target == s_tile_ptr[MAGMA_CSR5_THREAD_GROUP-1])
            {
                atomicAddmagmaDoubleComplex(&d_y[row_start_target], sum);
            }
            else
                d_y[row_start_target] += sum;
        }
    }
}

void 
spmv_csr5_tail_tile_kernel(
    const magma_index_t           *d_row_pointer,
    const magma_index_t           *d_column_index,
          magmaDoubleComplex      *d_value,
          magmaDoubleComplex      *d_x,
          magmaDoubleComplex      *d_y,
    const magma_index_t            tail_tile_start,
    const magma_index_t            p,
    const int                      sigma,
    const magmaDoubleComplex       alpha,
    const sycl::nd_item<3> &item_ct1,
    magmaDoubleComplex *s_sum)
{
    const int local_id = item_ct1.get_local_id(2);

    const magma_index_t row_id = tail_tile_start + item_ct1.get_group(2);
    const magma_index_t row_start = !item_ct1.get_group(2)
                                        ? (p - 1) * MAGMA_CSR5_OMEGA * sigma
                                        : d_row_pointer[row_id];
    const magma_index_t row_stop  = d_row_pointer[row_id + 1];

    magmaDoubleComplex sum = sycl::double2(r, i);

    for (magma_index_t idx = local_id + row_start; 
         idx < row_stop; idx += MAGMA_CSR5_OMEGA)
    {
        sum += candidate(d_value, d_x, d_column_index, idx, alpha);
    }
    s_sum[local_id] = sum;
    sum_32(s_sum, local_id);

    if (!local_id)
        d_y[row_id] += s_sum[0]; //= !blockIdx.x ? d_y[row_id] + sum : sum;
}

void 
zgecsr5mv_kernel_update_y(int    num_rows,
                          magmaDoubleComplex beta,
                          magmaDoubleComplex * dy,
                          const sycl::nd_item<3> &item_ct1)
{
    const magma_index_t row =
        item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);

    if (row < num_rows)
    {
        if (beta == sycl::double2(r, i))
            dy[row] = sycl::double2(r, i);
        else
            dy[row] *= beta; 
    }
}

/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    The input format is CSR5 (val (tile-wise column-major), 
                              row_pointer, 
                              col (tile-wise column-major),
                              tile_pointer, 
                              tile_desc).
    
    Arguments
    ---------
    
    @param[in]
    transA      magma_trans_t
                transposition parameter for A
                
    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 

    @param[in]
    p           magma_int_t
                number of tiles in A 

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    sigma       magma_int_t
                sigma in A in CSR5

    @param[in]
    bit_y_offset magma_int_t
                 bit_y_offset in A in CSR5

    @param[in]
    bit_scansum_offset  magma_int_t
                        bit_scansum_offset in A in CSR5

    @param[in]
    num_packet  magma_int_t
                num_packet in A in CSR5

    @param[in]
    dtile_ptr   magmaUIndex_ptr
                tilepointer of A in CSR5

    @param[in]
    dtile_desc  magmaUIndex_ptr
                tiledescriptor of A in CSR5

    @param[in]
    dtile_desc_offset_ptr  magmaIndex_ptr
                           tiledescriptor_offsetpointer of A in CSR5
                           
    @param[in]
    dtile_desc_offset      magmaIndex_ptr
                           tiledescriptor_offsetpointer of A in CSR5

    @param[in]
    dcalibrator  magmaDoubleComplex_ptr
                 calibrator of A in CSR5

    @param[in]
    tail_tile_start   magma_int_t
                      start of the last tile in A

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in CSR

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in CSR

    @param[in]
    drowptr     magmaIndex_ptr
                rowpointer of A in CSR

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in CSR

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier

    @param[out]
    dy          magmaDoubleComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zgecsr5mv(
    magma_trans_t           transA,
    magma_int_t             m, 
    magma_int_t             n, 
    magma_int_t             p,
    magmaDoubleComplex      alpha,
    magma_int_t             sigma,
    magma_int_t             bit_y_offset,
    magma_int_t             bit_scansum_offset,
    magma_int_t             num_packet,
    magmaUIndex_ptr         dtile_ptr,
    magmaUIndex_ptr         dtile_desc,
    magmaIndex_ptr          dtile_desc_offset_ptr,
    magmaIndex_ptr          dtile_desc_offset,
    magmaDoubleComplex_ptr  dcalibrator,
    magma_int_t             tail_tile_start,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          drowptr,
    magmaIndex_ptr          dcolind,
    magmaDoubleComplex_ptr  dx,
    magmaDoubleComplex      beta,
    magmaDoubleComplex_ptr  dy,
    magma_queue_t           queue )
{
    int info = MAGMA_ERR_NOT_SUPPORTED;
    
    // phase 1. update y: y = beta * y
    magma_int_t num_threads = MAGMA_CSR5_THREAD_GROUP;
    magma_int_t num_blocks = magma_ceildiv( m, num_threads ); 
    //ceil ((double)m / (double)num_threads);

    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, num_threads),
                              sycl::range<3>(1, 1, num_threads)),
            [=](sycl::nd_item<3> item_ct1) {
                zgecsr5mv_kernel_update_y(m, beta, dy, item_ct1);
            });

    // phase 2. spmv: y += alpha * A * x
    num_threads = MAGMA_CSR5_THREAD_GROUP;
    num_blocks = magma_ceildiv( p-1, num_threads / MAGMA_CSR5_OMEGA ); 
    // ceil ((double)(p-1) / (double)(num_threads / MAGMA_CSR5_OMEGA));

    switch (sigma)
    {
    case 4:
        /*
        DPCT1049:4: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<4>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 5:
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<5>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 6:
        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<6>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 7:
        /*
        DPCT1049:7: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<7>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 8:
        /*
        DPCT1049:8: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<8>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 9:
        /*
        DPCT1049:9: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<9>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 10:
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<10>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;

    case 11:
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<11>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 12:
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<12>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 13:
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<13>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 14:
        /*
        DPCT1049:14: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<14>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 15:
        /*
        DPCT1049:15: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<15>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 16:
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<16>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 17:
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<17>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 18:
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<18>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 19:
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<19>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 20:
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<20>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;

    case 21:
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<21>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 22:
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<22>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 23:
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<23>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 24:
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<24>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 25:
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                  MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<25>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 26:
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<26>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 27:
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<27>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 28:
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<28>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 29:
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<29>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 30:
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<30>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;

    case 31:
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                   MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<31>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    case 32:
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may
        exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if
        needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);
            sycl::local_accessor<int, 1>
                s_scan_acc_ct1(sycl::range<1>((MAGMA_CSR5_OMEGA + 1)
* (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)),
                               cgh);
            sycl::local_accessor<magma_uindex_t, 1> s_row_start_stop_acc_ct1(
                sycl::range<1>(
                    MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_compute_kernel<32>(
                        dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc,
                        dtile_desc_offset_ptr, dtile_desc_offset,
                        dcalibrator, dy, p, num_packet, bit_y_offset,
                        bit_scansum_offset, alpha, item_ct1,
                        s_sum_acc_ct1.get_pointer(),
                        s_scan_acc_ct1.get_pointer(),
                        s_row_start_stop_acc_ct1.get_pointer());
                });
        });
        break;
    }

    num_threads = MAGMA_CSR5_THREAD_GROUP;
    num_blocks = ceil((double)(p-1)/(double)num_threads);

    /*
    DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<volatile magma_index_t, 1>
                s_tile_ptr_acc_ct1(
                    sycl::range<1>(MAGMA_CSR5_THREAD_GROUP+1), cgh);
            sycl::local_accessor<magmaDoubleComplex, 1>
                s_calibrator_acc_ct1(
                    sycl::range<1>(MAGMA_CSR5_THREAD_GROUP), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_calibrate_kernel(
                        dtile_ptr, dcalibrator, dy, p, item_ct1,
                        s_tile_ptr_acc_ct1.get_pointer(),
                        s_calibrator_acc_ct1.get_pointer());
                });
        });

    num_threads = MAGMA_CSR5_OMEGA;
    num_blocks = m - tail_tile_start;

    /*
    DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<magmaDoubleComplex, 1> s_sum_acc_ct1(
                sycl::range<1>(MAGMA_CSR5_OMEGA), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, num_threads),
                                  sycl::range<3>(1, 1, num_threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    spmv_csr5_tail_tile_kernel(drowptr, dcolind, dval, dx,
                                               dy, tail_tile_start, p,
                                               sigma, alpha, item_ct1,
                                               s_sum_acc_ct1.get_pointer());
                });
        });

    info = MAGMA_SUCCESS;
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

    return info;
}
