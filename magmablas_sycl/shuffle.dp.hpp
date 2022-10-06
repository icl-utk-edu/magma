#ifndef ICL_MAGMA_SHUFFLE_CUH
#define ICL_MAGMA_SHUFFLE_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define SHFL_FULL_MASK 0xffffffff

// cuda 9.0 supports double precision shuffle
// but it is slower than the split by 32
// enable this flag to use the cuda version
//#define USE_CUDA_DP_SHFL
/******************************************************************************/
/**                       SHUFFLE BY INDEX                                   **/
/******************************************************************************/
static inline int magmablas_ishfl(int var, int srcLane,
                                  sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if DPCT_COMPATIBILITY_TEMP >= 300
#if __CUDACC_VER_MAJOR__ < 9 || defined(MAGMA_HAVE_HIP)
     return __shfl(var, srcLane, width);
#else
     /*
     DPCT1023:253: The DPC++ sub-group does not support mask options for
     dpct::select_from_sub_group.
     */
     return dpct::select_from_sub_group(item_ct1.get_sub_group(), var, srcLane,
                                        width);
#endif
#else    // pre-Kepler GPUs
return 0;
#endif
}


/******************************************************************************/
static inline float magmablas_sshfl(float var, int srcLane,
                                    sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if DPCT_COMPATIBILITY_TEMP >= 300
#if __CUDACC_VER_MAJOR__ < 9 || defined(MAGMA_HAVE_HIP)
     return __shfl(var, srcLane, width);
#else
     /*
     DPCT1023:254: The DPC++ sub-group does not support mask options for
     dpct::select_from_sub_group.
     */
     return dpct::select_from_sub_group(item_ct1.get_sub_group(), var, srcLane,
                                        width);
#endif
#else    // pre-Kepler GPUs
return MAGMA_S_ZERO;
#endif
}


/******************************************************************************/
static inline double magmablas_dshfl(double var, int srcLane,
                                     sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if DPCT_COMPATIBILITY_TEMP >= 300
#if __CUDACC_VER_MAJOR__ < 9 || defined(MAGMA_HAVE_HIP)
    // Split the double number into 2 32b registers.
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi):"d"(var));
    // Shuffle the two 32b registers.
    lo = __shfl(lo, srcLane, width);
    hi = __shfl(hi, srcLane, width);
    // Recreate the 64b number.
    asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
#else
    #ifdef USE_CUDA_DP_SHFL
    return __shfl_sync(mask, var, srcLane, width);
    #else
    // Split the double number into 2 32b registers.
    int lo, hi;
    /*
    DPCT1053:255: Migration of device assembly code is not supported.
    */
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
    // Shuffle the two 32b registers.
    /*
    DPCT1023:257: The DPC++ sub-group does not support mask options for
    dpct::select_from_sub_group.
    */
    /*
    DPCT1096:1664: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    lo = dpct::select_from_sub_group(item_ct1.get_sub_group(), lo, srcLane,
                                     width);
    /*
    DPCT1023:258: The DPC++ sub-group does not support mask options for
    dpct::select_from_sub_group.
    */
    /*
    DPCT1096:1665: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    hi = dpct::select_from_sub_group(item_ct1.get_sub_group(), hi, srcLane,
                                     width);
    // Recreate the 64b number.
    /*
    DPCT1053:256: Migration of device assembly code is not supported.
    */
    asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
    #endif
#endif
#else    // pre-Kepler GPUs
return MAGMA_D_ZERO;
#endif
}


/******************************************************************************/
static inline magmaFloatComplex magmablas_cshfl(magmaFloatComplex var, int srcLane,
                                                sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaFloatComplex r;
    r.real() = magmablas_sshfl(var.real(), srcLane, item_ct1, width, mask);
    r.imag() = magmablas_sshfl(var.imag(), srcLane, item_ct1, width, mask);
    return r;
}


/******************************************************************************/
static inline magmaDoubleComplex magmablas_zshfl(magmaDoubleComplex var, int srcLane,
                                                 sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaDoubleComplex r;
    r.real() = magmablas_dshfl(var.real(), srcLane, item_ct1, width, mask);
    r.imag() = magmablas_dshfl(var.imag(), srcLane, item_ct1, width, mask);
    return r;
}


/******************************************************************************/
/**                  SHUFFLE BY BITWISE XOR TO OWN LANE                      **/
/******************************************************************************/
static inline int magmablas_ishfl_xor(int var, int laneMask,
                                      sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if DPCT_COMPATIBILITY_TEMP >= 300
#if __CUDACC_VER_MAJOR__ < 9 || defined(MAGMA_HAVE_HIP)
    return __shfl_xor(var, laneMask, width);
#else
    /*
    DPCT1023:259: The DPC++ sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    return dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var,
                                          laneMask, width);
#endif
#else    // pre-Kepler GPUs
return 0;
#endif

}


/******************************************************************************/
static inline float magmablas_sshfl_xor(float var, int laneMask,
                                        sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if DPCT_COMPATIBILITY_TEMP >= 300
#if __CUDACC_VER_MAJOR__ < 9 || defined(MAGMA_HAVE_HIP)
    return __shfl_xor(var, laneMask, width);
#else
    /*
    DPCT1023:260: The DPC++ sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    return dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var,
                                          laneMask, width);
#endif
#else    // pre-Kepler GPUs
return MAGMA_S_ZERO;
#endif
}


/******************************************************************************/
static inline double magmablas_dshfl_xor(double var, int laneMask,
                                         sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if DPCT_COMPATIBILITY_TEMP >= 300
#if __CUDACC_VER_MAJOR__ < 9 || defined(MAGMA_HAVE_HIP)
    // Split the double number into 2 32b registers.
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi):"d"(var));
    // Shuffle the two 32b registers.
    lo = __shfl_xor(lo, laneMask, width);
    hi = __shfl_xor(hi, laneMask, width);
    // Recreate the 64b number.
    asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
#else
    #ifdef USE_CUDA_DP_SHFL
    return __shfl_xor_sync(mask, var, lanemask, width);
    #else
    // Split the double number into 2 32b registers.
    int lo, hi;
    /*
    DPCT1053:261: Migration of device assembly code is not supported.
    */
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
    // Shuffle the two 32b registers.
    /*
    DPCT1023:263: The DPC++ sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    lo = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), lo, laneMask,
                                        width);
    /*
    DPCT1023:264: The DPC++ sub-group does not support mask options for
    dpct::permute_sub_group_by_xor.
    */
    hi = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), hi, laneMask,
                                        width);
    // Recreate the 64b number.
    /*
    DPCT1053:262: Migration of device assembly code is not supported.
    */
    asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
    #endif
#endif
#else    // pre-Kepler GPUs
return MAGMA_D_ZERO;
#endif
}


/******************************************************************************/
static inline magmaFloatComplex magmablas_cshfl_xor(magmaFloatComplex var, int laneMask,
                                                    sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaFloatComplex r;
    r.real() = magmablas_sshfl_xor(var.real(), laneMask, item_ct1, width, mask);
    r.imag() = magmablas_sshfl_xor(var.imag(), laneMask, item_ct1, width, mask);
    return r;
}


/******************************************************************************/
static inline magmaDoubleComplex magmablas_zshfl_xor(magmaDoubleComplex var, int laneMask,
                                                     sycl::nd_item<3> item_ct1, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaDoubleComplex r;
    r.real() = magmablas_dshfl_xor(var.real(), laneMask, item_ct1, width, mask);
    r.imag() = magmablas_dshfl_xor(var.imag(), laneMask, item_ct1, width, mask);
    return r;
}


#endif    // ICL_MAGMA_SHUFFLE_CUH

