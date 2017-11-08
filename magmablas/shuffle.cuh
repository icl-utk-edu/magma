#ifndef ICL_MAGMA_SHUFFLE_CUH
#define ICL_MAGMA_SHUFFLE_CUH

#include "magma_internal.h"

#define SHFL_FULL_MASK 0xffffffff

// cuda 9.0 supports double precision shuffle
// but it is slower than the split by 32
// enable this flag to use the cuda version
//#define USE_CUDA_DP_SHFL
/******************************************************************************/
/**                       SHUFFLE BY INDEX                                   **/
/******************************************************************************/
__device__ static inline int magmablas_ishfl(int var, int srcLane, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
     return __shfl(var, srcLane, width);
#else
     return __shfl_sync(mask, var, srcLane, width);
#endif
#else    // pre-Kepler GPUs
return 0;
#endif
}


/******************************************************************************/
__device__ static inline float magmablas_sshfl(float var, int srcLane, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
     return __shfl(var, srcLane, width);
#else
     return __shfl_sync(mask, var, srcLane, width);
#endif
#else    // pre-Kepler GPUs
return MAGMA_S_ZERO;
#endif
}


/******************************************************************************/
__device__ static inline double magmablas_dshfl(double var, int srcLane, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
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
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi):"d"(var));
    // Shuffle the two 32b registers.
    lo = __shfl_sync(mask, lo, srcLane, width);
    hi = __shfl_sync(mask, hi, srcLane, width);
    // Recreate the 64b number.
    asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
    #endif
#endif
#else    // pre-Kepler GPUs
return MAGMA_D_ZERO;
#endif
}


/******************************************************************************/
__device__ static inline magmaFloatComplex magmablas_cshfl(magmaFloatComplex var, int srcLane, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaFloatComplex r; 
    r.x = magmablas_sshfl(var.x, srcLane, width, mask);
    r.y = magmablas_sshfl(var.y, srcLane, width, mask);
    return r;
}


/******************************************************************************/
__device__ static inline magmaDoubleComplex magmablas_zshfl(magmaDoubleComplex var, int srcLane, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaDoubleComplex r; 
    r.x = magmablas_dshfl(var.x, srcLane, width, mask);
    r.y = magmablas_dshfl(var.y, srcLane, width, mask);
    return r;
}


/******************************************************************************/
/**                  SHUFFLE BY BITWISE XOR TO OWN LANE                      **/
/******************************************************************************/
__device__ static inline int magmablas_ishfl_xor(int var, int laneMask, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
    return __shfl_xor(var, laneMask, width);
#else
    return __shfl_xor_sync(mask, var, laneMask, width);
#endif
#else    // pre-Kepler GPUs
return 0;
#endif

}


/******************************************************************************/
__device__ static inline float magmablas_sshfl_xor(float var, int laneMask, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
    return __shfl_xor(var, laneMask, width);
#else
    return __shfl_xor_sync(mask, var, laneMask, width);
#endif
#else    // pre-Kepler GPUs
return MAGMA_S_ZERO;
#endif
}


/******************************************************************************/
__device__ static inline double magmablas_dshfl_xor(double var, int laneMask, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
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
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi):"d"(var));
    // Shuffle the two 32b registers.
    lo = __shfl_xor_sync(mask, lo, laneMask, width);
    hi = __shfl_xor_sync(mask, hi, laneMask, width);
    // Recreate the 64b number.
    asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
    #endif
#endif
#else    // pre-Kepler GPUs
return MAGMA_D_ZERO;
#endif
}


/******************************************************************************/
__device__ static inline magmaFloatComplex magmablas_cshfl_xor(magmaFloatComplex var, int laneMask, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaFloatComplex r; 
    r.x = magmablas_sshfl_xor(var.x, laneMask, width, mask);
    r.y = magmablas_sshfl_xor(var.y, laneMask, width, mask);
    return r;
}


/******************************************************************************/
__device__ static inline magmaDoubleComplex magmablas_zshfl_xor(magmaDoubleComplex var, int laneMask, int width=32, unsigned mask=SHFL_FULL_MASK) 
{
    magmaDoubleComplex r; 
    r.x = magmablas_dshfl_xor(var.x, laneMask, width, mask);
    r.y = magmablas_dshfl_xor(var.y, laneMask, width, mask);
    return r;
}


#endif    // ICL_MAGMA_SHUFFLE_CUH

