/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Moritz Kreutzer

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#include "magmasparse_z.h"
#define BLOCK_SIZE 512

//#define PRECISION_z

  // for CUDA_VERSION

//#ifdef MAGMA_HAVE_CUDA
//#if (CUDA_VERSION <= 6000) && !defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)
//// CUDA 6.5 adds Double precision version; here's an implementation for CUDA 6.0 and earlier.
//// from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
//__device__ inline
//real_Double_t __shfl_down(real_Double_t var, unsigned int srcLane, int width=32) {
//  int2 a = *reinterpret_cast<int2*>(&var);
//  a.x = __shfl_down(a.x, srcLane, width);
//  a.y = __shfl_down(a.y, srcLane, width);
//  return *reinterpret_cast<double*>(&a);
//}
//#endif
//#endif


template<typename T>
__inline__ 
T warpReduceSum(T val, sycl::nd_item<3> item_ct1)
{
//#if DPCT_COMPATIBILITY_TEMP >= 300
//#if __CUDACC_VER_MAJOR__ < 9
//    val += __shfl_down(val, 16);
//    val += __shfl_down(val, 8);
//    val += __shfl_down(val, 4);
//    val += __shfl_down(val, 2);
//    val += __shfl_down(val, 1);
//#else
//    val += __shfl_down_sync(0xffffffff,val, 16);
//    val += __shfl_down_sync(0xffffffff,val, 8);
//    val += __shfl_down_sync(0xffffffff,val, 4);
//    val += __shfl_down_sync(0xffffffff,val, 2);
//    val += __shfl_down_sync(0xffffffff,val, 1);
//#endif
//#endif
  auto subgroup = item_ct1.get_sub_group();
  val = sycl::reduce_over_group(subgroup, val, sycl::plus<>());

    return val;
}


//#ifdef PRECISION_z
//template<>
//__inline__ 
//magmaDoubleComplex warpReduceSum<magmaDoubleComplex>(magmaDoubleComplex val,
//                                                     sycl::nd_item<3> item_ct1)
//{
//#ifdef MAGMA_HAVE_CUDA
//#if __CUDACC_VER_MAJOR__ < 9
//    a.x += __shfl_down(a.x, 16);
//    a.y += __shfl_down(a.y, 16);
//    a.z += __shfl_down(a.z, 16);
//    a.w += __shfl_down(a.w, 16);
//    a.x += __shfl_down(a.x, 8);
//    a.y += __shfl_down(a.y, 8);
//    a.z += __shfl_down(a.z, 8);
//    a.w += __shfl_down(a.w, 8);
//    a.x += __shfl_down(a.x, 4);
//    a.y += __shfl_down(a.y, 4);
//    a.z += __shfl_down(a.z, 4);
//    a.w += __shfl_down(a.w, 4);
//    a.x += __shfl_down(a.x, 2);
//    a.y += __shfl_down(a.y, 2);
//    a.z += __shfl_down(a.z, 2);
//    a.w += __shfl_down(a.w, 2);
//    a.x += __shfl_down(a.x, 1);
//    a.y += __shfl_down(a.y, 1);
//    a.z += __shfl_down(a.z, 1);
//    a.w += __shfl_down(a.w, 1);
//#endif
//#endif
//#if DPCT_COMPATIBILITY_TEMP >= 300
//    sycl::int4 a = *reinterpret_cast<sycl::int4 *>(&val);
//    /*
//    DPCT1023:804: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:844: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.x() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.x(), 16);
//    /*
//    DPCT1023:805: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:845: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.y() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.y(), 16);
//    /*
//    DPCT1023:806: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:846: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.z() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.z(), 16);
//    /*
//    DPCT1023:807: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:847: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.w() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.w(), 16);
//    /*
//    DPCT1023:808: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:848: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.x() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.x(), 8);
//    /*
//    DPCT1023:809: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:849: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.y() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.y(), 8);
//    /*
//    DPCT1023:810: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:850: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.z() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.z(), 8);
//    /*
//    DPCT1023:811: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:851: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.w() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.w(), 8);
//    /*
//    DPCT1023:812: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:852: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.x() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.x(), 4);
//    /*
//    DPCT1023:813: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:853: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.y() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.y(), 4);
//    /*
//    DPCT1023:814: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:854: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.z() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.z(), 4);
//    /*
//    DPCT1023:815: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:855: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.w() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.w(), 4);
//    /*
//    DPCT1023:816: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:856: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.x() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.x(), 2);
//    /*
//    DPCT1023:817: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:857: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.y() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.y(), 2);
//    /*
//    DPCT1023:818: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:858: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.z() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.z(), 2);
//    /*
//    DPCT1023:819: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:859: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.w() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.w(), 2);
//    /*
//    DPCT1023:820: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:860: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.x() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.x(), 1);
//    /*
//    DPCT1023:821: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:861: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.y() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.y(), 1);
//    /*
//    DPCT1023:822: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:862: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.z() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.z(), 1);
//    /*
//    DPCT1023:823: The DPC++ sub-group does not support mask options for
//    dpct::shift_sub_group_left.
//    */
//    /*
//    DPCT1096:863: The right-most dimension of the work-group used in the SYCL
//    kernel that calls this function may be less than "32". The function
//    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
//    device. Modify the size of the work-group to ensure that the value of the
//    right-most dimension is a multiple of "32".
//    */
//    a.w() += dpct::shift_sub_group_left(item_ct1.get_sub_group(), a.w(), 1);
//#endif
//    return val;
//}
//#endif // PRECISION_z


//#ifdef PRECISION_c
//template<>
//__inline__
//magmaFloatComplex warpReduceSum<magmaFloatComplex>(magmaFloatComplex val)
//{
//#if __CUDA_ARCH__ >= 300
//    float2 a = *reinterpret_cast<float2*>(&val);
//#if __CUDACC_VER_MAJOR__ < 9
//    a.x += __shfl_down(a.x, 16);
//    a.y += __shfl_down(a.y, 16);
//    a.x += __shfl_down(a.x, 8);
//    a.y += __shfl_down(a.y, 8);
//    a.x += __shfl_down(a.x, 4);
//    a.y += __shfl_down(a.y, 4);
//    a.x += __shfl_down(a.x, 2);
//    a.y += __shfl_down(a.y, 2);
//    a.x += __shfl_down(a.x, 1);
//    a.y += __shfl_down(a.y, 1);
//#else
//    a.x += __shfl_down_sync(0xffffffff,a.x, 16);
//    a.y += __shfl_down_sync(0xffffffff,a.y, 16);
//    a.x += __shfl_down_sync(0xffffffff,a.x, 8);
//    a.y += __shfl_down_sync(0xffffffff,a.y, 8);
//    a.x += __shfl_down_sync(0xffffffff,a.x, 4);
//    a.y += __shfl_down_sync(0xffffffff,a.y, 4);
//    a.x += __shfl_down_sync(0xffffffff,a.x, 2);
//    a.y += __shfl_down_sync(0xffffffff,a.y, 2);
//    a.x += __shfl_down_sync(0xffffffff,a.x, 1);
//    a.y += __shfl_down_sync(0xffffffff,a.y, 1);
//#endif
//#endif
//    return val;
//}
//#endif // PRECISION_c


template<typename T>
__inline__ 
T blockReduceSum_1D(T val, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared = (T *)dpct_local; // Shared mem for 32 partial sums
    int lane = item_ct1.get_local_id(2) %
               item_ct1.get_sub_group().get_local_range().get(0);
    int wid = item_ct1.get_local_id(2) /
              item_ct1.get_sub_group().get_local_range().get(0);

    val =
        warpReduceSum<T>(val, item_ct1); // Each warp performs partial reduction

    if (lane == 0) shared[wid]=val; // Write reduced value to shared memory

    /*
    DPCT1065:824: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    /*
    DPCT1064:825: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    val = (item_ct1.get_local_id(2) <
           item_ct1.get_local_range(2) /
               item_ct1.get_sub_group().get_local_range().get(0))
              ? shared[lane]
              : MAGMA_Z_ZERO;

    if (wid == 0) val =
        warpReduceSum<T>(val, item_ct1); // Final reduce within first warp
    return val;
}


template<typename T>
__inline__ 
T blockReduceSum(T val, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared = (T *)dpct_local; // Shared mem for 32 partial sums
    int lane = item_ct1.get_local_id(2) %
               item_ct1.get_sub_group().get_local_range().get(0);
    int wid = item_ct1.get_local_id(2) /
              item_ct1.get_sub_group().get_local_range().get(0);

    val =
        warpReduceSum<T>(val, item_ct1); // Each warp performs partial reduction

    if (lane == 0) shared[item_ct1.get_local_id(1) * 32 + wid] =
        val; // Write reduced value to shared memory

    /*
    DPCT1065:826: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    /*
    DPCT1064:827: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    val = (item_ct1.get_local_id(2) <
           item_ct1.get_local_range(2) /
               item_ct1.get_sub_group().get_local_range().get(0))
              ? shared[item_ct1.get_local_id(1) * 32 + lane]
              : MAGMA_Z_ZERO;

    if (wid == 0) val =
        warpReduceSum<T>(val, item_ct1); // Final reduce within first warp
    return val;
}


template<typename T> 
void deviceReduceKernel(const T * __restrict__ in, T * __restrict__ out, int N,
                        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    T sum = 0.0;
    //reduce multiple elements per thread
    for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         i < N;
         i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sum += in[i];
    }
    sum = blockReduceSum<T>(sum, item_ct1, dpct_local);
    if (item_ct1.get_local_id(2) == 0)
        out[item_ct1.get_group(2)] = sum;
}


// dot product for multiple vectors using shuffle intrinsics and less shared memory
void
magma_zblockdot_kernel_shuffle( 
    int n, 
    int k,
    const magmaDoubleComplex * __restrict__ v,
    const magmaDoubleComplex * __restrict__ r,
    magmaDoubleComplex * __restrict__ vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_local_id(1);
    magmaDoubleComplex tmp;
    if (i < n) {
        tmp = v[i+j*n] * r[i];
    } else {
        /*
        DPCT1064:828: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        tmp = MAGMA_Z_ZERO;
    }
    tmp = blockReduceSum(tmp, item_ct1, dpct_local);
    if (item_ct1.get_local_id(2) == 0) {
        vtmp[item_ct1.get_group(2) + j * item_ct1.get_group_range(2)] = tmp;
    }
}


// dot product for multiple vectors using shuffle intrinsics and less shared memory
void
magma_zblockdot_kernel_shuffle_1dblock( 
    int n, 
    int k,
    const magmaDoubleComplex * __restrict__ v,
    const magmaDoubleComplex * __restrict__ r,
    magmaDoubleComplex * __restrict__ vtmp,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int j;
    for (j=0; j < k; j++) {
        magmaDoubleComplex tmp;
        if (i < n) {
            tmp = v[i+j*n] * r[i];
        } else {
            tmp = MAGMA_Z_ZERO;
        }
        tmp = blockReduceSum_1D(tmp, item_ct1, dpct_local);
        if (item_ct1.get_local_id(2) == 0) {
            vtmp[item_ct1.get_group(2) + j * item_ct1.get_group_range(2)] = tmp;
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
        sycl::range<3> block(1, 1, BLOCK_SIZE);
        sycl::range<3> grid(1, 1, magma_ceildiv(n, block[2]));
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                /*
                DPCT1083:843: The size of local memory in the migrated code may
                be different from the original code. Check that the allocated
                memory size in the migrated code is correct.
                */
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(
                        sycl::range<1>(32 * sizeof(magmaDoubleComplex)), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                32)]] {
                        magma_zblockdot_kernel_shuffle_1dblock(
                            n, k, v, r, d1, item_ct1,
                            (uint8_t *)dpct_local_acc_ct1.get_pointer());
                    });
            });
        int j;
        for (j=0; j < k; j++) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1083:864: The size of local memory in the migrated code
                    may be different from the original code. Check that the
                    allocated memory size in the migrated code is correct.
                    */
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(
                            sycl::range<1>(32 * sizeof(magmaDoubleComplex)),
                            cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_SIZE),
                                          sycl::range<3>(1, 1, BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[intel::reqd_sub_group_size(32)]] {
                                deviceReduceKernel<magmaDoubleComplex>(
                                    d1 + grid[2] * j, skp + j, grid[2],
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                });
        }
    } else {
        sycl::range<3> block(1, k,
                             magma_roundup(magma_ceildiv(BLOCK_SIZE, k), 32));
        int max_wg_size = queue->sycl_stream()->get_device()
                            .get_info<sycl::info::device::max_work_group_size>(); 
        while (block[2] * block[1] > max_wg_size) {
            block[2] -= 32;
        }
        sycl::range<3> grid(1, 1, magma_ceildiv(n, block[2]));
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                /*
                DPCT1083:865: The size of local memory in the migrated code may
                be different from the original code. Check that the allocated
                memory size in the migrated code is correct.
                */
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(
                        sycl::range<1>(32 * k * sizeof(magmaDoubleComplex)),
                        cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block), [=
                ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                                                32)]] {
                        magma_zblockdot_kernel_shuffle(
                            n, k, v, r, d1, item_ct1,
                            (uint8_t *)dpct_local_acc_ct1.get_pointer());
                    });
            });
        int j;
        for (j=0; j < k; j++) {
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1083:866: The size of local memory in the migrated code
                    may be different from the original code. Check that the
                    allocated memory size in the migrated code is correct.
                    */
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(
                            sycl::range<1>(32 * sizeof(magmaDoubleComplex)),
                            cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_SIZE),
                                          sycl::range<3>(1, 1, BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[intel::reqd_sub_group_size(32)]] {
                                deviceReduceKernel<magmaDoubleComplex>(
                                    d1 + grid[2] * j, skp + j, grid[2],
                                    item_ct1, dpct_local_acc_ct1.get_pointer());
                            });
                });
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
