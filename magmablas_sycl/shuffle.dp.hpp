#ifndef ICL_MAGMA_SHUFFLE_CUH
#define ICL_MAGMA_SHUFFLE_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define SHFL_FULL_MASK 0xffffffff
#define DEFAULT_WIDTH  32
typedef unsigned shfl_mask_t;

/******************************************************************************/
/**                       SHUFFLE BY INDEX                                   **/
/******************************************************************************/
static inline int magmablas_ishfl(int var, int srcLane,
                                  const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
     /*
     DPCT1108:131: '__shfl_sync' was migrated with the experimental feature
     masked sub_group function which may not be supported by all compilers or
     runtimes. You may need to adjust the code.
     */
     return dpct::experimental::select_from_sub_group(
         mask, item_ct1.get_sub_group(), var, srcLane, width);
}


/******************************************************************************/
static inline float magmablas_sshfl(float var, int srcLane,
                                    const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
     /*
     DPCT1108:132: '__shfl_sync' was migrated with the experimental feature
     masked sub_group function which may not be supported by all compilers or
     runtimes. You may need to adjust the code.
     */
     return dpct::experimental::select_from_sub_group(
         mask, item_ct1.get_sub_group(), var, srcLane, width);
}


/******************************************************************************/
static inline double magmablas_dshfl(double var, int srcLane,
                                     const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
     /*
     DPCT1108:132: '__shfl_sync' was migrated with the experimental feature
     masked sub_group function which may not be supported by all compilers or
     runtimes. You may need to adjust the code.
     */
     return dpct::experimental::select_from_sub_group(
         mask, item_ct1.get_sub_group(), var, srcLane, width);
}


/******************************************************************************/
static inline magmaFloatComplex magmablas_cshfl(magmaFloatComplex var, int srcLane,
                                                const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    float r_real, r_imag;
    r_real = magmablas_sshfl(MAGMA_C_REAL(var), srcLane, item_ct1, width, mask);
    r_imag = magmablas_sshfl(MAGMA_C_IMAG(var), srcLane, item_ct1, width, mask);
    magmaFloatComplex r = MAGMA_C_MAKE(r_real, r_imag);
    return r;
}


/******************************************************************************/
static inline magmaDoubleComplex magmablas_zshfl(magmaDoubleComplex var, int srcLane,
                                                 const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    double r_real, r_imag;
    r_real = magmablas_dshfl(MAGMA_Z_REAL(var), srcLane, item_ct1, width, mask);
    r_imag = magmablas_dshfl(MAGMA_Z_IMAG(var), srcLane, item_ct1, width, mask);
    magmaDoubleComplex r = MAGMA_Z_MAKE(r_real, r_imag);
    return r;
}


/******************************************************************************/
/**                  SHUFFLE BY BITWISE XOR TO OWN LANE                      **/
/******************************************************************************/
static inline int magmablas_ishfl_xor(int var, int laneMask,
                                      const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    /*
    DPCT1108:137: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    return dpct::experimental::permute_sub_group_by_xor(
        mask, item_ct1.get_sub_group(), var, laneMask, width);
}


/******************************************************************************/
static inline float magmablas_sshfl_xor(float var, int laneMask,
                                        const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    /*
    DPCT1108:138: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    return dpct::experimental::permute_sub_group_by_xor(
        mask, item_ct1.get_sub_group(), var, laneMask, width);
}


/******************************************************************************/
static inline double magmablas_dshfl_xor(double var, int laneMask,
                                         const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    /*
    DPCT1108:138: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    return dpct::experimental::permute_sub_group_by_xor(
        mask, item_ct1.get_sub_group(), var, laneMask, width);
}


/******************************************************************************/
static inline magmaFloatComplex magmablas_cshfl_xor(magmaFloatComplex var, int laneMask,
                                                    const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    float r_real, r_imag;
    r_real = magmablas_sshfl_xor(MAGMA_C_REAL(var), laneMask, item_ct1, width, mask);
    r_imag = magmablas_sshfl_xor(MAGMA_C_IMAG(var), laneMask, item_ct1, width, mask);
    magmaFloatComplex r = MAGMA_C_MAKE(r_real, r_imag);
    return r;
}


/******************************************************************************/
static inline magmaDoubleComplex magmablas_zshfl_xor(magmaDoubleComplex var, int laneMask,
                                                     const sycl::nd_item<3> &item_ct1, int width=DEFAULT_WIDTH, shfl_mask_t mask=SHFL_FULL_MASK)
{
    double r_real, r_imag;
    r_real = magmablas_dshfl_xor(MAGMA_Z_REAL(var), laneMask, item_ct1, width, mask);
    r_imag = magmablas_dshfl_xor(MAGMA_Z_IMAG(var), laneMask, item_ct1, width, mask);
    magmaDoubleComplex r = MAGMA_Z_MAKE(r_real, r_imag);
    return r;
}

#endif    // ICL_MAGMA_SHUFFLE_DPCPP

