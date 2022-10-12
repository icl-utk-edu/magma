/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       [zcds]gemm_fermi.cu        defines the CPU driver.
       [zcds]gemm_fermi_kernels.h defines the block sizes for each precision.
       [zcds]gemm_fermi_kernels_batched.h defines the block sizes for each precision.
       gemm_stencil_defs.h        defines types and functions for precision-independent code.
       gemm_stencil.cu            defines the GPU kernel. It gets included
                                  multiple times, once for each transpose version.
*/


// =============================================================================

#ifdef COMPLEX
  #ifdef DOUBLE
    typedef magmaDoubleComplex FloatingPoint_t;
    #define precision z
  #else
    typedef magmaFloatComplex FloatingPoint_t;
    #define precision c
  #endif
#else
  #ifdef DOUBLE
    typedef double FloatingPoint_t;
    #define precision d
  #else
    typedef float FloatingPoint_t;
    #define precision s
  #endif
#endif

// =============================================================================
#define fetch(A, m, n, bound) offs_d##A[min(n*LD##A+m, bound)]

#ifdef COMPLEX
  #ifdef DOUBLE
    #define conj(A)          MAGMA_Z_CONJ(A)
    #define add(A, B)        MAGMA_Z_ADD(A, B)
    #define mul(A, B)        MAGMA_Z_MUL(A, B)
    #define fma(A, B, C) C = magmaCfma(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_Z_MAKE(x, y);
  #else
    #define conj(A)          MAGMA_C_CONJ(A)
    #define add(A, B)        MAGMA_C_ADD(A, B)
    #define mul(A, B)        MAGMA_C_MUL(A, B)
    #define fma(A, B, C) C = magmaCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_C_MAKE(x, y);
  #endif
#else
    #define conj(A)           (A)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

// =============================================================================

#define trans_nn 1
#define trans_nt 2
#define trans_nc 3
        
#define trans_tn 4
#define trans_tt 5
#define trans_tc 6
        
#define trans_cn 7
#define trans_ct 8
#define trans_cc 9
