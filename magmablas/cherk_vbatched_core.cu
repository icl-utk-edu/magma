/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_c

#include "herk_template_kernel_vbatched.cuh"
#include "gemm_config/cgemm_param_nn.h"
#include "gemm_config/cgemm_param_nt.h"
#include "gemm_config/cgemm_param_tn.h"
#include "gemm_config/cgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
template<int CONJ>
void
magmablas_csyrkherk_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmaFloatComplex cbeta  = beta;
    magmaFloatComplex calpha = alpha;
    
    // we have two shapes
    magma_int_t shape;
    if      (trans == MagmaNoTrans)   { shape = 0; } // nc or nt
    else                              { shape = 1; } // cn or tn
        
    switch(shape)
    {
        case 0: // nc
            {
                if(max_k < 64)
                {
                    herk_template_vbatched_nt<magmaFloatComplex, version(NT,338), 0, CONJ>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
                else
                {
                    if(max_n < 128)
                    {
                        herk_template_vbatched_nt<magmaFloatComplex, version(NT,338), 0, CONJ>
                        (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                    }
                    else
                    {
                        herk_template_vbatched_nt<magmaFloatComplex, version(NT,426), 0, CONJ>
                        (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                    }
                }
            }
            break;
        case 1: // cn
            {
                if(max_k < 16)
                {
                    herk_template_vbatched_tn<magmaFloatComplex, version(TN,282), CONJ, 0>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
                else
                {
                    herk_template_vbatched_tn<magmaFloatComplex, version(TN,505), CONJ, 0>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
            }
            break;
        default:; // propose something
    }
}


/******************************************************************************/
extern "C" void
magmablas_csyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmablas_csyrkherk_vbatched<0>(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}


/******************************************************************************/
extern "C" void
magmablas_cherk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmablas_csyrkherk_vbatched<1>(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}


/******************************************************************************/
extern "C" void
magmablas_csyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue )
{
    magmablas_csyrk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dA_array, ldda, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}


/******************************************************************************/
extern "C" void
magmablas_cherk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue )
{
    magmablas_cherk_internal_vbatched(uplo, trans, n, k, MAGMA_C_MAKE(alpha, 0.), dA_array, ldda, dA_array, ldda, MAGMA_C_MAKE(beta, 0.), dC_array, lddc, max_n, max_k, batchCount, queue );
}
