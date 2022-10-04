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

#define PRECISION_z

#include "gemm_template_kernel_vbatched.dp.hpp"

#include "gemm_config/zgemm_param_nn.h"
#include "gemm_config/zgemm_param_nt.h"
#include "gemm_config/zgemm_param_tn.h"
#include "gemm_config/zgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_zgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    if(max_m <=0 || max_n <= 0 || max_k <= 0) return;

    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc

    switch(shape)
    {
        case 0: // nn
            {
                gemm_template_vbatched_nn<magmaDoubleComplex, version(NN,18), 0, 0>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 1: // nt
            {
                if(max_k <= 8)
                {
                    // version 58
                    gemm_template_vbatched_nt<magmaDoubleComplex, version(NT,58), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    // version 29
                    gemm_template_vbatched_nt<magmaDoubleComplex, version(NT,29), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 2: // nc
            {
                if(max_k <= 8)
                {
                    // version 58
                    gemm_template_vbatched_nt<magmaDoubleComplex, version(NT,58), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    // version 29
                    gemm_template_vbatched_nt<magmaDoubleComplex, version(NT,29), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 3: // tn
            {
                // version 72
                gemm_template_vbatched_tn<magmaDoubleComplex, version(TN,72), 0, 0>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 6: // cn
            {
                // version 72
                gemm_template_vbatched_tn<magmaDoubleComplex, version(TN,72), 1, 0>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 4: // tt
            {
                // version 13
                gemm_template_vbatched_tt<magmaDoubleComplex, version(TT,13), 0, 0>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 5: // tc
            {
                // version 13
                gemm_template_vbatched_tt<magmaDoubleComplex, version(TT,13), 0, 1>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 7: // ct
            {
                // version 13
                gemm_template_vbatched_tt<magmaDoubleComplex, version(TT,13), 1, 0>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 8: // cc
            {
                // version 13
                gemm_template_vbatched_tt<magmaDoubleComplex, version(TT,13), 1, 1>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        default:; // propose something
    }
}
