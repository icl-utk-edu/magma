/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Moritz Kreutzer

*/
#include "magmasparse_internal.h"

#include "magmasparse_z.h"
#define BLOCK_SIZE 512

#define PRECISION_z

#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION <= 6000)
// CUDA 6.5 adds Double precision version; here's an implementation for CUDA 6.0 and earlier.
// from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__device__ inline
real_Double_t __shfl_down(real_Double_t var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}
#endif


template<typename T>
__inline__ __device__
T warpReduceSum(T val)
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
#else
    val += __shfl_down_sync(0xffffffff,val, 16);
    val += __shfl_down_sync(0xffffffff,val, 8);
    val += __shfl_down_sync(0xffffffff,val, 4);
    val += __shfl_down_sync(0xffffffff,val, 2);
    val += __shfl_down_sync(0xffffffff,val, 1);
#endif
#endif
    return val;
}


#ifdef PRECISION_z
template<>
__inline__ __device__
magmaDoubleComplex warpReduceSum<magmaDoubleComplex>(magmaDoubleComplex val)
{
#if __CUDA_ARCH__ >= 300
    int4 a = *reinterpret_cast<int4*>(&val);
#if __CUDACC_VER_MAJOR__ < 9
    a.x += __shfl_down(a.x, 16);
    a.y += __shfl_down(a.y, 16);
    a.z += __shfl_down(a.z, 16);
    a.w += __shfl_down(a.w, 16);
    a.x += __shfl_down(a.x, 8);
    a.y += __shfl_down(a.y, 8);
    a.z += __shfl_down(a.z, 8);
    a.w += __shfl_down(a.w, 8);
    a.x += __shfl_down(a.x, 4);
    a.y += __shfl_down(a.y, 4);
    a.z += __shfl_down(a.z, 4);
    a.w += __shfl_down(a.w, 4);
    a.x += __shfl_down(a.x, 2);
    a.y += __shfl_down(a.y, 2);
    a.z += __shfl_down(a.z, 2);
    a.w += __shfl_down(a.w, 2);
    a.x += __shfl_down(a.x, 1);
    a.y += __shfl_down(a.y, 1);
    a.z += __shfl_down(a.z, 1);
    a.w += __shfl_down(a.w, 1);
#else
    a.x += __shfl_down_sync(0xffffffff,a.x, 16);
    a.y += __shfl_down_sync(0xffffffff,a.y, 16);
    a.z += __shfl_down_sync(0xffffffff,a.z, 16);
    a.w += __shfl_down_sync(0xffffffff,a.w, 16);
    a.x += __shfl_down_sync(0xffffffff,a.x, 8);
    a.y += __shfl_down_sync(0xffffffff,a.y, 8);
    a.z += __shfl_down_sync(0xffffffff,a.z, 8);
    a.w += __shfl_down_sync(0xffffffff,a.w, 8);
    a.x += __shfl_down_sync(0xffffffff,a.x, 4);
    a.y += __shfl_down_sync(0xffffffff,a.y, 4);
    a.z += __shfl_down_sync(0xffffffff,a.z, 4);
    a.w += __shfl_down_sync(0xffffffff,a.w, 4);
    a.x += __shfl_down_sync(0xffffffff,a.x, 2);
    a.y += __shfl_down_sync(0xffffffff,a.y, 2);
    a.z += __shfl_down_sync(0xffffffff,a.z, 2);
    a.w += __shfl_down_sync(0xffffffff,a.w, 2);
    a.x += __shfl_down_sync(0xffffffff,a.x, 1);
    a.y += __shfl_down_sync(0xffffffff,a.y, 1);
    a.z += __shfl_down_sync(0xffffffff,a.z, 1);
    a.w += __shfl_down_sync(0xffffffff,a.w, 1);
#endif
#endif
    return val;
}
#endif // PRECISION_z


#ifdef PRECISION_c
template<>
__inline__ __device__
magmaFloatComplex warpReduceSum<magmaFloatComplex>(magmaFloatComplex val)
{
#if __CUDA_ARCH__ >= 300
    float2 a = *reinterpret_cast<float2*>(&val);
#if __CUDACC_VER_MAJOR__ < 9
    a.x += __shfl_down(a.x, 16);
    a.y += __shfl_down(a.y, 16);
    a.x += __shfl_down(a.x, 8);
    a.y += __shfl_down(a.y, 8);
    a.x += __shfl_down(a.x, 4);
    a.y += __shfl_down(a.y, 4);
    a.x += __shfl_down(a.x, 2);
    a.y += __shfl_down(a.y, 2);
    a.x += __shfl_down(a.x, 1);
    a.y += __shfl_down(a.y, 1);
#else
    a.x += __shfl_down_sync(0xffffffff,a.x, 16);
    a.y += __shfl_down_sync(0xffffffff,a.y, 16);
    a.x += __shfl_down_sync(0xffffffff,a.x, 8);
    a.y += __shfl_down_sync(0xffffffff,a.y, 8);
    a.x += __shfl_down_sync(0xffffffff,a.x, 4);
    a.y += __shfl_down_sync(0xffffffff,a.y, 4);
    a.x += __shfl_down_sync(0xffffffff,a.x, 2);
    a.y += __shfl_down_sync(0xffffffff,a.y, 2);
    a.x += __shfl_down_sync(0xffffffff,a.x, 1);
    a.y += __shfl_down_sync(0xffffffff,a.y, 1);
#endif
#endif
    return val;
}
#endif // PRECISION_c


template<typename T>
__inline__ __device__
T blockReduceSum_1D(T val)
{
    extern __shared__ T shared[]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum<T>(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : MAGMA_Z_ZERO;
    
    if (wid == 0) val = warpReduceSum<T>(val); //Final reduce within first warp
    return val;
}


template<typename T>
__inline__ __device__
T blockReduceSum(T val)
{
    extern __shared__ T shared[]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum<T>(val);     // Each warp performs partial reduction

    if (lane == 0) shared[threadIdx.y*32+wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.y*32+lane] : MAGMA_Z_ZERO;
    
    if (wid == 0) val = warpReduceSum<T>(val); //Final reduce within first warp
    return val;
}


template<typename T> 
__global__ void deviceReduceKernel(const T * __restrict__ in, T * __restrict__ out, int N)
{
    T sum = MAGMA_Z_MAKE(0.0, 0.0);
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum<T>(sum);
    if (threadIdx.x == 0)
        out[blockIdx.x]=sum;
}


// dot product for multiple vectors using shuffle intrinsics and less shared memory
__global__ void
magma_zblockdot_kernel_shuffle( 
    int n, 
    int k,
    const magmaDoubleComplex * __restrict__ v,
    const magmaDoubleComplex * __restrict__ r,
    magmaDoubleComplex * __restrict__ vtmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    magmaDoubleComplex tmp;
    if (i < n) {
        tmp = v[i+j*n] * r[i];
    } else {
        tmp = MAGMA_Z_ZERO;
    }
    tmp = blockReduceSum(tmp);
    if (threadIdx.x == 0 ){
        vtmp[ blockIdx.x+j*gridDim.x ] = tmp;
    }
}


// dot product for multiple vectors using shuffle intrinsics and less shared memory
__global__ void
magma_zblockdot_kernel_shuffle_1dblock( 
    int n, 
    int k,
    const magmaDoubleComplex * __restrict__ v,
    const magmaDoubleComplex * __restrict__ r,
    magmaDoubleComplex * __restrict__ vtmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    for (j=0; j < k; j++) {
        magmaDoubleComplex tmp;
        if (i < n) {
            tmp = v[i+j*n] * r[i];
        } else {
            tmp = MAGMA_Z_ZERO;
        }
        tmp = blockReduceSum_1D(tmp);
        if (threadIdx.x == 0 ){
            vtmp[ blockIdx.x+j*gridDim.x ] = tmp;
        }
    }
}


/**
    Purpose
    -------

    Computes the scalar product of a set of vectors v_i such that

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDoubleComplex_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDoubleComplex_ptr 
                r

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    if ( magma_getdevice_arch() < 300 ) {
        return magma_zmdotc( n, k, v, r, d1, d2, skp, queue );
    }
    else if (1) { // 1D block kernel seems to be always faster
        dim3 block( BLOCK_SIZE );
        dim3 grid( magma_ceildiv( n, block.x ) );
        magma_zblockdot_kernel_shuffle_1dblock<<< grid, block, 32*sizeof(magmaDoubleComplex), queue->cuda_stream() >>>( n, k, v, r, d1 );
        int j;
        for (j=0; j < k; j++) {
            deviceReduceKernel<magmaDoubleComplex> <<<1, 1024, 32*sizeof(magmaDoubleComplex), queue->cuda_stream()>>>(d1+grid.x*j, skp+j, grid.x);
        }
    } else {
        dim3 block( magma_roundup( magma_ceildiv(BLOCK_SIZE, k), 32 ), k );
        while (block.x*block.y > 1024) {
            block.x -= 32;
        }
        dim3 grid( magma_ceildiv( n, block.x ) );
        magma_zblockdot_kernel_shuffle<<< grid, block, 32*k*sizeof(magmaDoubleComplex), queue->cuda_stream() >>>( n, k, v, r, d1 );
        int j;
        for (j=0; j < k; j++) {
            deviceReduceKernel<magmaDoubleComplex> <<<1, 1024, 32*sizeof(magmaDoubleComplex), queue->cuda_stream()>>>(d1+grid.x*j, skp+j, grid.x);
        }
    }
   
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    This is an extension of the merged dot product above by chunking
    the set of vectors v_i such that the data always fits into cache.
    It is equivalent to a matrix vecor product Vr where V
    contains few rows and many columns. The computation is the same:

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDoubleComplex_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDoubleComplex_ptr 
                r

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zgemvmdot_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    if (k == 1) { // call CUBLAS dotc, we will never be faster
        magmaDoubleComplex res = magma_zdotc( n, v, 1, r, 1, queue );
        magma_zsetvector( 1, &res, 1, skp, 1, queue );
    }
    else if ( magma_getdevice_arch() < 300 ) {
        return magma_zgemvmdot( n, k, v, r, d1, d2, skp, queue );
    }
    else {
        magma_zmdotc_shfl( n, k, v, r, d1, d2, skp, queue );
    }

    return MAGMA_SUCCESS;
}
