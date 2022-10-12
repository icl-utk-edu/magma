/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Ahmad Abdelfattah
       @author Azzam Haidar

*/

#ifndef GEMM_TEMPLATE_DEVICE_DEFS_H
#define GEMM_TEMPLATE_DEVICE_DEFS_H

// =============================================================================
#define fetch(A, m, n, bound) offs_d##A[min(n*LD##A+m, bound)]


#if defined(PRECISION_z)
    #define conj(A)          MAGMA_Z_CONJ(A)
    #define add(A, B)        MAGMA_Z_ADD(A, B)
    #define mul(A, B)        MAGMA_Z_MUL(A, B)
    #define div(A, B)        MAGMA_Z_DIV(A, B)
    #define fma(A, B, C) C = magmaCfma(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_Z_MAKE(x, y)
#elif defined(PRECISION_c)
    #define conj(A)          MAGMA_C_CONJ(A)
    #define add(A, B)        MAGMA_C_ADD(A, B)
    #define mul(A, B)        MAGMA_C_MUL(A, B)
    #define div(A, B)        MAGMA_C_DIV(A, B)
    #define fma(A, B, C) C = magmaCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_C_MAKE(x, y)
#elif defined(PRECISION_h)
    #define conj(A)           (A)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) ((magmaHalf)x)
#else
    #define conj(A)           (A)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

#if defined(PRECISION_z)
    #define magmablas_atomic_add magmablas_zatomic_add
#elif defined(PRECISION_c)
    #define magmablas_atomic_add magmablas_catomic_add
#elif defined(PRECISION_d)
    #define magmablas_atomic_add magmablas_datomic_add
#else
    #define magmablas_atomic_add magmablas_satomic_add
#endif

#endif // GEMM_TEMPLATE_DEVICE_DEFS_H
