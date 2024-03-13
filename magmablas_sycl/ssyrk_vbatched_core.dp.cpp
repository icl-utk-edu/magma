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
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define PRECISION_s

#include "herk_template_kernel_vbatched.dp.hpp"

#include "gemm_config/sgemm_param_nn.h"
#include "gemm_config/sgemm_param_nt.h"
#include "gemm_config/sgemm_param_tn.h"
#include "gemm_config/sgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_ssyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    float cbeta  = MAGMA_S_MAKE( beta, 0. );
    float calpha = MAGMA_S_MAKE( alpha, 0. );
    
    // we have two shapes only (nt or tn)
    magma_int_t shape;
    if      (trans == MagmaNoTrans)   { shape = 0; } // nt
    else                              { shape = 1; } // tn
    
    switch(shape)
    {
        case 0: // nt
            {
                herk_template_vbatched_nt<float, version(NT,734), 0, 0>
                (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
            }
            break;
        case 1: // tn
            {
                if(max_k < 64)
                {
                    herk_template_vbatched_tn<float, version(TN,654), 0, 0>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
                else
                {
                    herk_template_vbatched_tn<float, version(TN,666), 0, 0>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
            }
            break;
        default:; // propose something
    }
}


/******************************************************************************/
extern "C" void
magmablas_ssyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float beta,
    float **dC_array, magma_int_t* lddc,  
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue )
{
    magmablas_ssyrk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dA_array, ldda, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}
