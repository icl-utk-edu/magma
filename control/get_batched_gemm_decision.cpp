/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Definition of blocking sizes for NVIDIA cards
#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)

// =============================================================================
/// @addtogroup magma_tuning
/// @{

// auxiliary function to determine the limit of use for the specialized batch gemm
// kernel on small square sizes (transpositions do not matter for this kernel)
// helper function - intended for internal use only
magma_int_t magma_get_zgemm_batched_smallsq_limit(magma_int_t n)
{
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) return 22;
    else if (arch <= 600) return 28;
    else if (arch <= 700) return 27;
    else                  return 16;

}

/******************************************************************************/
/// @see magma_get_zgemm_batched_smallsq_limit
magma_int_t magma_get_cgemm_batched_smallsq_limit(magma_int_t n)
{
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) return 22;
    else if (arch <= 600) return 20;
    else if (arch <= 700) return 20;
    else                  return 16;

}

/******************************************************************************/
/// @see magma_get_zgemm_batched_smallsq_limit
magma_int_t magma_get_dgemm_batched_smallsq_limit(magma_int_t n)
{
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) return 23;
    else if (arch <= 600) return 23;
    else if (arch <= 700) return 22;
    else                  return 16;

}

/******************************************************************************/
/// @see magma_get_zgemm_batched_smallsq_limit
magma_int_t magma_get_sgemm_batched_smallsq_limit(magma_int_t n)
{
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) return 29;
    else if (arch <= 600) return 31;
    else if (arch <= 700) return 27;
    else                  return 16;

}

// =============================================================================
/// @addtogroup magma_tuning
/// @{

// Advisory functions used to determine if cuBLAS should be used for batched gemm
// Decision is based on the dimensions and the shape
// Two cuBLAS-based alternatives are used (batched vs streamed)
// Decisions are based on tuning experiments conducted on Kepler K40c (bunsen).

// helper function - intended for internal use only
magma_int_t magma_get_gemm_shape(magma_trans_t transA, magma_trans_t transB)
{
    magma_int_t shape = -1;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc

    return shape;
}


/***************************************************************************//**
    Decides which is better (magma or cublas_batched),
    regardless of the performance of cublas stream

    @return true  (1) to use cuBLAS batched gemm
    @return false (0) to use MAGMA  batched gemm
*******************************************************************************/
magma_int_t magma_srecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
        {
            use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && ( !(  k == 32             ) ) // ! k == 32
                                                        );
        }
        break;

        case 1: // nt
        case 2: // nc
        {
            use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                   && ( !(  k == 32             ) ) // ! k == 32
                                                        );
        }
            break;

        case 3: // tn
        case 6: // cn
        {
            use_cublas_gemm_batched = 1;
            if(m == n && m <= 32) {
                #ifdef MAGMA_HAVE_CUDA
                switch(n) {
                    case 15:
                    case 16: use_cublas_gemm_batched = 0; break;
                    case  2: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case  3: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case  4: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case  5: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case  6: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case  7: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case  8: use_cublas_gemm_batched = (k >  600) ? 1: 0; break;
                    case  9: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 10: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 11: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case 12: use_cublas_gemm_batched = (k >  600) ? 1: 0; break;
                    case 13: use_cublas_gemm_batched = (k > 1200) ? 1: 0; break;
                    case 14: use_cublas_gemm_batched = (k > 1200) ? 1: 0; break;
                    case 17: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 18: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    case 19: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 20: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 21: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 22: use_cublas_gemm_batched = (k > 1500) ? 1: 0; break;
                    case 23: use_cublas_gemm_batched = (k > 1500) ? 1: 0; break;
                    case 24: use_cublas_gemm_batched = (k > 1300) ? 1: 0; break;
                    case 25: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 26: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 27: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 28: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 29: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 30: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 31: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 32: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #else
                switch(n) {
                    case  2:
                    case  3:
                    case  4:
                    case  5:
                    case  6:
                    case  7:
                    case  8:
                    case  9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16:
                    case 19: use_cublas_gemm_batched = 0; break;
                    case 17: use_cublas_gemm_batched = (k > 1500) ? 1: 0; break;
                    case 18: use_cublas_gemm_batched = (k > 1600) ? 1: 0; break;
                    case 20: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    case 21: use_cublas_gemm_batched = (k > 1800) ? 1: 0; break;
                    case 22: use_cublas_gemm_batched = (k > 1500) ? 1: 0; break;
                    case 23: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case 24: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    case 25: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case 26: use_cublas_gemm_batched = (k > 1300) ? 1: 0; break;
                    case 27: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 28: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    case 29: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 30: use_cublas_gemm_batched = (k > 1100) ? 1: 0; break;
                    case 31: use_cublas_gemm_batched = (k >  900) ? 1: 0; break;
                    case 32: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #endif  // MAGMA_HAVE_CUDA
            } // (m == n && m <= 32)
        }
        break;

        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
        {
            use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && ( !(  k == 32             ) ) // ! k == 32
                                                        );
        }
        break;

        default:;
    }
    return use_cublas_gemm_batched;
}


/******************************************************************************/
/// @see magma_srecommend_cublas_gemm_batched
magma_int_t magma_drecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        // fall back to cublas, since it is very optimized in general
        // TODO: revisit tuning for other shapes, especially small sizes
        case 0: // nn
        case 1: // nt
        case 2: // nc
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
        {
            use_cublas_gemm_batched = 1;
        }
        break;
        case 3: // tn
        case 6: // cn
        {
            use_cublas_gemm_batched = 1;
            if(m == n && m <= 32) {
                #ifdef MAGMA_HAVE_CUDA
                switch(n) {
                    case  7:
                    case  8: use_cublas_gemm_batched = 0; break;
                    case 32: use_cublas_gemm_batched = 1; break;
                    case  2: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case  3: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case  4: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case  5: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case  6: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case  9: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 10: use_cublas_gemm_batched = (k > 1200) ? 1: 0; break;
                    case 11: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case 12: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case 13: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case 14: use_cublas_gemm_batched = (k > 1600) ? 1: 0; break;
                    case 15: use_cublas_gemm_batched = (k > 1600) ? 1: 0; break;
                    case 16: use_cublas_gemm_batched = (k > 1600) ? 1: 0; break;
                    case 17: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case 18: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 19: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case 20: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 21: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 22: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case 23: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case 24: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case 25: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 26: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 27: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case 28: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case 29: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case 30: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case 31: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #else
                switch(n) {
                    case  2:
                    case  3:
                    case  4:
                    case  5:
                    case  6:
                    case  7:
                    case  8:
                    case  9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16:
                    case 17: use_cublas_gemm_batched = 0; break;
                    case 18: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    case 19: use_cublas_gemm_batched = (k > 1600) ? 1: 0; break;
                    case 20: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 21: use_cublas_gemm_batched = (k > 1900) ? 1: 0; break;
                    case 22: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 23: use_cublas_gemm_batched = (k > 1700) ? 1: 0; break;
                    case 24: use_cublas_gemm_batched = (k > 1800) ? 1: 0; break;
                    case 25: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 26: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 27: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 28: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 29: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 30: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 31: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 32: use_cublas_gemm_batched = (k >  600) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #endif  // MAGMA_HAVE_CUDA
            } // (m == n && m <= 32)
        }

        default:;
    }
    return use_cublas_gemm_batched;
}


/******************************************************************************/
/// @see magma_srecommend_cublas_gemm_batched
magma_int_t magma_crecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
        {
            use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                   && (  (  k < 32 )              ) // k < 32
                                                    );
        }
        break;

        case 1: // nt
        case 2: // nc
        {
            // No cublas batched for this case
            use_cublas_gemm_batched = 0;
        }
        break;

        case 3: // tn
        case 6: // cn
        {
            use_cublas_gemm_batched = 1;
            if(m == n && m <= 32) {
                #ifdef MAGMA_HAVE_CUDA
                switch(n) {
                    case  2:
                    case  3:
                    case  4:
                    case  5:
                    case  6:
                    case  7:
                    case  8:
                    case  9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 24: use_cublas_gemm_batched = 0; break;
                    case 25: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case 26: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 27: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 28: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case 29: use_cublas_gemm_batched = (k >  900) ? 1: 0; break;
                    case 30: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case 31: use_cublas_gemm_batched = (k >  900) ? 1: 0; break;
                    case 32: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #else
                switch(n) {
                    case  2:
                    case  3:
                    case  4:
                    case  5:
                    case  6:
                    case  7:
                    case  8:
                    case  9:
                    case 10:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 16: use_cublas_gemm_batched = 0; break;
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                    case 23:
                    case 24:
                    case 25:
                    case 26:
                    case 27:
                    case 28:
                    case 29:
                    case 30:
                    case 31:
                    case 32: use_cublas_gemm_batched = 1; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #endif  // MAGMA_HAVE_CUDA
            }  // m == n && m <= 32
        }
        break;

        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
        {
            use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                   && (  (  k < 32 )              ) // k < 32
                                                    );
        }
        break;

        default:;
    }
    return use_cublas_gemm_batched;
}


/******************************************************************************/
/// @see magma_srecommend_cublas_gemm_batched
magma_int_t magma_zrecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
        case 1: // nt
        case 2: // nc
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
        {
            use_cublas_gemm_batched = 1;
        }
        break;

        case 3: // tn
        case 6: // cn
        {
            use_cublas_gemm_batched = 1;
            if(m == n && m <= 32) {
                #ifdef MAGMA_HAVE_CUDA
                switch(n) {
                    case  2:
                    case  3:
                    case  4:
                    case  5:
                    case  6:
                    case  7:
                    case  8: use_cublas_gemm_batched = 0; break;
                    case  9: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case 10: use_cublas_gemm_batched = (k >  900) ? 1: 0; break;
                    case 11: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case 12: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case 13: use_cublas_gemm_batched = (k > 1200) ? 1: 0; break;
                    case 14: use_cublas_gemm_batched = (k > 1400) ? 1: 0; break;
                    case 15: use_cublas_gemm_batched = (k > 1500) ? 1: 0; break;
                    case 16: use_cublas_gemm_batched = (k > 1600) ? 1: 0; break;
                    case 17: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 18: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 19: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 20: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 21: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case 22: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case 23: use_cublas_gemm_batched = (k >  600) ? 1: 0; break;
                    case 24: use_cublas_gemm_batched = (k >  600) ? 1: 0; break;
                    case 25: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 26: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 27: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 28: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 29: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 30: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 31: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    case 32: use_cublas_gemm_batched = (k >  100) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #else
                switch(n) {
                    case 25:
                    case 26:
                    case 27:
                    case 28:
                    case 29:
                    case 30:
                    case 31: use_cublas_gemm_batched = 1; break;
                    case  2: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case  3: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case  4: use_cublas_gemm_batched = (k >  800) ? 1: 0; break;
                    case  5: use_cublas_gemm_batched = (k >  600) ? 1: 0; break;
                    case  6: use_cublas_gemm_batched = (k > 1100) ? 1: 0; break;
                    case  7: use_cublas_gemm_batched = (k >  700) ? 1: 0; break;
                    case  8: use_cublas_gemm_batched = (k > 1000) ? 1: 0; break;
                    case  9: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 10: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 11: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 12: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 13: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 14: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 15: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 16: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    case 17: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 18: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 19: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 20: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 21: use_cublas_gemm_batched = (k >  500) ? 1: 0; break;
                    case 22: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 23: use_cublas_gemm_batched = (k >  300) ? 1: 0; break;
                    case 24: use_cublas_gemm_batched = (k >  400) ? 1: 0; break;
                    case 32: use_cublas_gemm_batched = (k >  200) ? 1: 0; break;
                    default: use_cublas_gemm_batched = 1;
                }
                #endif  // MAGMA_HAVE_CUDA
            }  // m == n && m <= 32
        }
        break;

        default:;
    }
    return use_cublas_gemm_batched;
}


/***************************************************************************//**
    Decides if cublas stream should be used for a given gemm dimension/shape

    @return true  (1) to use cuBLAS gemm (non-batched) with multiple streams.
    @return false (0) to use batched gemm
*******************************************************************************/
magma_int_t magma_srecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 224         ) ) // k == 64, m >= 224
                                                      || ( (k >= 128            ) && (m >= 160         ) ) // k >= 128
                                                        );
            }
            break;

        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >=  224 && m < 512) ) // k == 64, m == 224:512
                                                      || ( (k >= 128            ) && (m >= 224            ) ) // k >= 128, m >= 224
                                                        );
            }
            break;

        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 192 && m <  512) ) // k == 64, m == 192:512
                                                      || ( (k >= 128            ) && (m >= 128            ) ) // k >= 128, m >= 224
                                                        );
            }
            break;

        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 192 && m <  512) ) // k == 64, m == 192:512
                                                      || ( (k >= 128            ) && (m >= 160            ) ) // k >= 128, m >= 224
                                                        );
            }
            break;

        default:;
    }
    return use_cublas_gemm_stream;
}


/******************************************************************************/
/// @see magma_srecommend_cublas_gemm_stream
magma_int_t magma_drecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 192         ) ) // k == 64, m >= 192
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 160         ) ) // k == 64, m >= 160
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 256         ) ) // k == 32, m >= 256
                                                      || ( (k >  32  && k <= 64 ) && (m >= 192         ) ) // k == 64, m >= 192
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 160         ) ) // k == 64, m >= 160
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        default:;
    }
    return use_cublas_gemm_stream;
}


/******************************************************************************/
/// @see magma_srecommend_cublas_gemm_stream
magma_int_t magma_crecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        default:;
    }
    return use_cublas_gemm_stream;
}


/******************************************************************************/
/// @see magma_srecommend_cublas_gemm_stream
magma_int_t magma_zrecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);

    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 160         ) ) // k == 32, m >= 160
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 64         ) ) // k >= 128, m >= 64
                                                        );
            }
            break;

        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 128         ) ) // k == 32, m >= 128
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 128         ) ) // k == 32, m >= 128
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 128         ) ) // k == 32, m >= 128
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;

        default:;
    }
    return use_cublas_gemm_stream;
}


// =============================================================================
/// @}
// end group magma_tuning

#endif  // MAGMA_HAVE_CUDA

#ifdef __cplusplus
} // extern "C"
#endif
