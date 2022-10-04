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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_z

#include "herk_template_kernel_vbatched.dp.hpp"
#include "gemm_config/zgemm_param_nn.h"
#include "gemm_config/zgemm_param_nt.h"
#include "gemm_config/zgemm_param_tn.h"
#include "gemm_config/zgemm_param_tt.h"

/******************************************************************************/
#define version(s,v) s ## _V_ ## v
template<int CONJ>
void
magmablas_zsyrkherk_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmaDoubleComplex cbeta  = beta;
    magmaDoubleComplex calpha = alpha;

    // we have two shapes only
    magma_int_t shape;
    if   (trans == MagmaNoTrans) { shape = 0; } // nc or nt
    else                         { shape = 1; } // cn or tn
        
    switch(shape)
    {
        case 0: // nc or nt
            {
                if(max_k <= 8)
                {
                    // version 58
                    herk_template_vbatched_nt<magmaDoubleComplex, version(NT,58), 0, CONJ>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
                else
                {
                    // version 29
                    herk_template_vbatched_nt<magmaDoubleComplex, version(NT,29), 0, CONJ>
                    (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
                }
            }
            break;
        case 1: // cn or tn
            {
                // version 72
                herk_template_vbatched_tn<magmaDoubleComplex, version(TN,72), CONJ, 0>
                (uplo, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, calpha, cbeta, batchCount, queue, max_n);
            }
            break;
        default:; // propose something
    }
}


/******************************************************************************/
extern "C" void
magmablas_zherk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmablas_zsyrkherk_vbatched<1>(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}


/******************************************************************************/
extern "C" void
magmablas_zsyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmablas_zsyrkherk_vbatched<0>(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}


/******************************************************************************/
extern "C" void 
magmablas_zsyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue )
{
    magmablas_zsyrk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dA_array, ldda, beta, dC_array, lddc, max_n, max_k, batchCount, queue );
}


/******************************************************************************/
extern "C" void 
magmablas_zherk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t* ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue )
{
    /*
    DPCT1064:1071: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmablas_zherk_internal_vbatched(
        uplo, trans, n, k, MAGMA_Z_MAKE(alpha, 0.), dA_array, ldda, dA_array,
        ldda, MAGMA_Z_MAKE(beta, 0.), dC_array, lddc, max_n, max_k, batchCount,
        queue);
}
