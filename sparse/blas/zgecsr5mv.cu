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

#include "magmasparse_internal.h"
#include "atomicopsmagmaDoubleComplex.h"

#include <cuda.h>  // for CUDA_VERSION

#define MAGMA_CSR5_THREAD_GROUP 128
#define MAGMA_CSR5_THREAD_BUNCH 32

#if (defined( CUDA_VERSION ) && ( CUDA_VERSION >= 8000 ))

__inline__ __device__ void
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

__inline__ __device__ void
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
    if (local_id == 0)  { s_scan[31] = s_scan[15]; s_scan[15] = MAGMA_Z_ZERO; }
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

__inline__ __device__ magmaDoubleComplex
candidate(
          magmaDoubleComplex      *d_value_tile,
          magmaDoubleComplex      *d_x,
    const magma_index_t           *d_column_index_tile,
    const magma_index_t            candidate_index,
    const magmaDoubleComplex       alpha)
{
    magmaDoubleComplex x = MAGMA_Z_ZERO;
#if __CUDA_ARCH__ >= 350
    x = __ldg(&d_x[d_column_index_tile[candidate_index]]);
#else
    x = d_x[d_column_index_tile[candidate_index]];
#endif
    return d_value_tile[candidate_index] * x * alpha;
}

//template<typename vT>
//__forceinline__ __device__
//vT segmented_sum_shfl(vT        tmp_sum,
//                      const int scansum_offset,
//                      const int lane_id)
//{
//    vT sum = __shfl_down(tmp_sum, 1);
//    sum = lane_id == MAGMA_CSR5_OMEGA - 1 ? 0 : sum;
//    // inclusive scan
//    vT scan_sum = scan_32_shfl(sum); //scan_32_shfl<vT>(sum, lane_id); 
//    tmp_sum = __shfl_down(scan_sum, scansum_offset);
//    tmp_sum = tmp_sum - scan_sum + sum;
//
//    return tmp_sum;
//}

__forceinline__ __device__ magmaDoubleComplex
segmented_sum(
             magmaDoubleComplex    tmp_sum,
             magmaDoubleComplex   *s_sum,
    const    magma_index_t         scansum_offset,
    const    magma_index_t         lane_id)
{
    if (lane_id)
        s_sum[lane_id - 1] = tmp_sum;
    s_sum[lane_id] = lane_id == MAGMA_CSR5_OMEGA - 1 
                     ? MAGMA_Z_ZERO : s_sum[lane_id];
    magmaDoubleComplex sum = tmp_sum = s_sum[lane_id];
    scan_32(s_sum, lane_id); // exclusive scan
    s_sum[lane_id] += tmp_sum; // inclusive scan (exclusive scan+original val)
    tmp_sum = s_sum[lane_id + scansum_offset];
    tmp_sum = tmp_sum - s_sum[lane_id] + sum;

    return tmp_sum;
}

template<int c_sigma>
__inline__ __device__ void 
tile_fast_track(
          magmaDoubleComplex    *d_value_tile,
          magmaDoubleComplex    *d_x,
    const magma_index_t         *d_column_index_tile,
          magmaDoubleComplex    *d_calibrator,
//#if __CUDA_ARCH__ < 300
          magmaDoubleComplex    *s_sum,
//#endif
    const int                    lane_id,
    const magma_index_t          par_id,
    const magmaDoubleComplex     alpha)
{
    magmaDoubleComplex sum = MAGMA_Z_ZERO;

    #pragma unroll
    for (int i = 0; i < c_sigma; i++)
    {
        sum += candidate(d_value_tile, d_x, d_column_index_tile, 
                         i * MAGMA_CSR5_OMEGA + lane_id, alpha);
    }

//#if __CUDA_ARCH__ >= 300 // use shfl intrinsic
//    sum = sum_32_shfl<vT>(sum);
//    if (!lane_id)
//        d_calibrator[par_id] = sum;
//#else // use smem
    s_sum[lane_id] = sum;
    sum_32(s_sum, lane_id);
    if (!lane_id)
    {
        d_calibrator[par_id] = s_sum[0];
    }
//#endif
}

template<int c_sigma>
__inline__ __device__ void 
tile_normal_track(
    const magma_index_t           *d_column_index_tile,
          magmaDoubleComplex      *d_value_tile,
          magmaDoubleComplex      *d_x,
    const magma_uindex_t          *d_tile_desc,
    const magma_index_t           *d_tile_desc_offset_ptr,
    const magma_index_t           *d_tile_desc_offset,
          magmaDoubleComplex      *d_calibrator,
          magmaDoubleComplex      *d_y,
//#if __CUDA_ARCH__ < 300
          magmaDoubleComplex      *s_sum,
    volatile int                  *s_scan,
//#endif
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
    magmaDoubleComplex sum  = MAGMA_Z_ZERO;

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
        sum = local_bit ? MAGMA_Z_ZERO : sum;
        stop += local_bit;

        sum += candidate(d_value_tile, d_x, d_column_index_tile, 
                         i * MAGMA_CSR5_OMEGA + lane_id, alpha);
    }

    first_sum = direct ? first_sum : sum;
    last_sum = sum;

    // step 2. segmented sum
    sum = start ? first_sum : MAGMA_Z_ZERO;

//#if __CUDA_ARCH__ >= 300
//    sum = segmented_sum_shfl<vT>(sum, scansum_offset, lane_id);
//#else
    sum = segmented_sum(sum, s_sum, scansum_offset, lane_id);
//#endif

    // step 3-1. add s_sum to position stop
    last_sum += (start <= stop) ? sum : MAGMA_Z_ZERO;

    // step 3-2. write sums to result array
    if (direct)
        d_y[empty_rows ? d_tile_desc_offset[offset_pointer + y_offset] 
                       : y_offset] += last_sum;

    // the first/last value of the first thread goes to calibration
    if (!lane_id)
        d_calibrator[par_id] = direct ? first_sum : last_sum;
}

template<int c_sigma>
__inline__ __device__ void 
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
    const magmaDoubleComplex       alpha)
{
//#if __CUDA_ARCH__ < 300
    __shared__ magmaDoubleComplex  
             s_sum[MAGMA_CSR5_THREAD_GROUP];
    volatile __shared__ int 
             s_scan[(MAGMA_CSR5_OMEGA + 1) * 
                    (MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA)];
//#endif

    magma_uindex_t row_start, row_stop;

//#if __CUDA_ARCH__ >= 350
//    if (lane_id < 2)
//        row_start = __ldg(&d_tile_ptr[par_id + lane_id]);
//    row_stop = __shfl(row_start, 1);
//    row_start = __shfl(row_start, 0);
//    row_stop &= 0x7FFFFFFF;
//#else
    volatile __shared__ magma_uindex_t 
             s_row_start_stop[MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1];
    if (threadIdx.x < MAGMA_CSR5_THREAD_GROUP / MAGMA_CSR5_OMEGA + 1)
    {
        s_row_start_stop[threadIdx.x] = d_tile_ptr[par_id + threadIdx.x];
    }
    __syncthreads();

    row_start = s_row_start_stop[bunch_id];
    row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;
//#endif

    if (row_start == row_stop) // fast track through reduction
    {
        tile_fast_track<c_sigma>
                (d_value_tile, d_x, d_column_index_tile, d_calibrator,
//#if __CUDA_ARCH__ < 300
                 &s_sum[bunch_id * MAGMA_CSR5_OMEGA],
//#endif
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
//#if __CUDA_ARCH__ < 300
                 &s_sum[bunch_id * MAGMA_CSR5_OMEGA],
                 &s_scan[bunch_id * (MAGMA_CSR5_OMEGA + 1)],
//#endif
                 par_id, lane_id,
                 bit_y_offset, bit_scansum_offset, empty_rows, alpha);
    }
}

template<int c_sigma>
__global__ void 
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
    const magmaDoubleComplex       alpha)
{
    // warp lane id
    const int lane_id = 31 & threadIdx.x; //threadIdx.x % CSR5_OMEGA;
    // warp global id == par_id
    const magma_index_t  par_id = (blockIdx.x * blockDim.x + threadIdx.x)
                                  / MAGMA_CSR5_OMEGA;
    const int bunch_id = threadIdx.x / MAGMA_CSR5_OMEGA;

    if (par_id >= p - 1)
        return;

    spmv_tile<c_sigma>
        (&d_column_index[par_id * MAGMA_CSR5_OMEGA * c_sigma],
         &d_value[par_id * MAGMA_CSR5_OMEGA * c_sigma],
         d_row_pointer, d_x, d_tile_ptr,
         &d_tile_desc[par_id * MAGMA_CSR5_OMEGA * num_packet],
         d_tile_desc_offset_ptr, d_tile_desc_offset,
         d_calibrator, d_y,
         par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, alpha);
}

__global__ void 
spmv_csr5_calibrate_kernel(
    const magma_uindex_t      *d_tile_ptr,
    const magmaDoubleComplex  *d_calibrator,
          magmaDoubleComplex  *d_y,
    const magma_index_t        p)
{
    //const int lane_id  = threadIdx.x % MAGMA_CSR5_THREAD_BUNCH;
    //const int bunch_id = threadIdx.x / MAGMA_CSR5_THREAD_BUNCH;
    const int local_id = threadIdx.x;
    const magma_index_t global_id = blockIdx.x * blockDim.x + threadIdx.x; 

    magmaDoubleComplex sum;

    volatile __shared__ 
             magma_index_t s_tile_ptr[MAGMA_CSR5_THREAD_GROUP+1];
    __shared__ magmaDoubleComplex  s_calibrator[MAGMA_CSR5_THREAD_GROUP];
    //volatile __shared__ 
    //         magmaDoubleComplex  s_sum[MAGMA_CSR5_THREAD_GROUP 
    //                                   / MAGMA_CSR5_THREAD_BUNCH];

    s_tile_ptr[local_id] = global_id < p-1 ? 
                  (magma_index_t)(d_tile_ptr[global_id] & 0x7FFFFFFF) : -1;
    s_calibrator[local_id] = sum = global_id < p-1 ? 
                                   d_calibrator[global_id] : MAGMA_Z_ZERO;
    __syncthreads();

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
        __syncthreads(); 
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
    sum = MAGMA_Z_ZERO;

    // use (p - 1), due to the tail tile is dealt with CSR-vector method
    if (global_id < p - 1)
    {
        row_start_previous = local_id ? s_tile_ptr[local_id-1] : -1;
        row_start_current = s_tile_ptr[local_id];

        if (row_start_previous != row_start_current)
        {
            row_start_target = row_start_current;

            while (row_start_target == row_start_current 
                   && local_par_id < blockDim.x)
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

__global__ void 
spmv_csr5_tail_tile_kernel(
    const magma_index_t           *d_row_pointer,
    const magma_index_t           *d_column_index,
          magmaDoubleComplex      *d_value,
          magmaDoubleComplex      *d_x,
          magmaDoubleComplex      *d_y,
    const magma_index_t            tail_tile_start,
    const magma_index_t            p,
    const int                      sigma,
    const magmaDoubleComplex       alpha)
{
    const int local_id = threadIdx.x;

    const magma_index_t row_id    = tail_tile_start + blockIdx.x;
    const magma_index_t row_start = !blockIdx.x ? (p - 1) 
                                    * MAGMA_CSR5_OMEGA * sigma 
                                    : d_row_pointer[row_id];
    const magma_index_t row_stop  = d_row_pointer[row_id + 1];

    magmaDoubleComplex sum = MAGMA_Z_ZERO;

    for (magma_index_t idx = local_id + row_start; 
         idx < row_stop; idx += MAGMA_CSR5_OMEGA)
    {
        sum += candidate(d_value, d_x, d_column_index, idx, alpha);
    }
//#if __CUDA_ARCH__ >= 300 // use shfl intrinsic
//    sum = sum_32_shfl<vT>(sum);
//#else
    __shared__ magmaDoubleComplex s_sum[MAGMA_CSR5_OMEGA];
    s_sum[local_id] = sum;
    sum_32(s_sum, local_id);
//#endif

    if (!local_id)
        d_y[row_id] += s_sum[0]; //= !blockIdx.x ? d_y[row_id] + sum : sum;
}

__global__ void 
zgecsr5mv_kernel_update_y(int    num_rows,
                          magmaDoubleComplex beta,
                          magmaDoubleComplex * dy)
{
    const magma_index_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows)
    {
        if (beta == MAGMA_Z_ZERO)
            dy[row] = MAGMA_Z_ZERO;
        else
            dy[row] *= beta; 
    }
}

#endif

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
    
#if (defined( CUDA_VERSION ) && ( CUDA_VERSION >= 8000 ))
    magma_int_t arch = magma_getdevice_arch();    
    if ( arch >= 600 ) {
        //dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
        //magma_int_t threads = BLOCK_SIZE;
        //zgecsrmv_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
        //                (m, n, alpha, dval, drowptr, dcolind, dx, beta, dy);
    
        // phase 1. update y: y = beta * y
        magma_int_t num_threads = MAGMA_CSR5_THREAD_GROUP;
        magma_int_t num_blocks = magma_ceildiv( m, num_threads ); 
        //ceil ((double)m / (double)num_threads);
    
        zgecsr5mv_kernel_update_y
            <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>(m, beta, dy);
    
        // phase 2. spmv: y += alpha * A * x
        num_threads = MAGMA_CSR5_THREAD_GROUP;
        num_blocks = magma_ceildiv( p-1, num_threads / MAGMA_CSR5_OMEGA ); 
        // ceil ((double)(p-1) / (double)(num_threads / MAGMA_CSR5_OMEGA));
    
        switch (sigma)
        {
        case 4:
            spmv_csr5_compute_kernel<4>
                 <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 5:
            spmv_csr5_compute_kernel<5>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 6:
            spmv_csr5_compute_kernel<6>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 7:
            spmv_csr5_compute_kernel<7>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 8:
            spmv_csr5_compute_kernel<8>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 9:
            spmv_csr5_compute_kernel<9>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 10:
            spmv_csr5_compute_kernel<10>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
    
        case 11:
            spmv_csr5_compute_kernel<11>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 12:
            spmv_csr5_compute_kernel<12>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 13:
            spmv_csr5_compute_kernel<13>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 14:
            spmv_csr5_compute_kernel<14>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 15:
            spmv_csr5_compute_kernel<15>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 16:
            spmv_csr5_compute_kernel<16>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 17:
            spmv_csr5_compute_kernel<17>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 18:
            spmv_csr5_compute_kernel<18>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 19:
            spmv_csr5_compute_kernel<19>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 20:
            spmv_csr5_compute_kernel<20>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
    
        case 21:
            spmv_csr5_compute_kernel<21>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 22:
            spmv_csr5_compute_kernel<22>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 23:
            spmv_csr5_compute_kernel<23>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 24:
            spmv_csr5_compute_kernel<24>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 25:
            spmv_csr5_compute_kernel<25>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 26:
            spmv_csr5_compute_kernel<26>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 27:
            spmv_csr5_compute_kernel<27>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 28:
            spmv_csr5_compute_kernel<28>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 29:
            spmv_csr5_compute_kernel<29>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 30:
            spmv_csr5_compute_kernel<30>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
    
        case 31:
            spmv_csr5_compute_kernel<31>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        case 32:
            spmv_csr5_compute_kernel<32>
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dcolind, dval, drowptr, dx, dtile_ptr, dtile_desc, 
                 dtile_desc_offset_ptr, dtile_desc_offset, dcalibrator, dy, p, 
                 num_packet, bit_y_offset, bit_scansum_offset, alpha);
            break;
        }
    
        num_threads = MAGMA_CSR5_THREAD_GROUP;
        num_blocks = ceil((double)(p-1)/(double)num_threads);
    
        spmv_csr5_calibrate_kernel
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (dtile_ptr, dcalibrator, dy, p);
    
        num_threads = MAGMA_CSR5_OMEGA;
        num_blocks = m - tail_tile_start;
    
        spmv_csr5_tail_tile_kernel
                <<< num_blocks, num_threads, 0, queue->cuda_stream() >>>
                (drowptr, dcolind, dval, dx, dy,
                 tail_tile_start, p, sigma, alpha);
                
        info = MAGMA_SUCCESS;
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
#endif

    return info;
}
