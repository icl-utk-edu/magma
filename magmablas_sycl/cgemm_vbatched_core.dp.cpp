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

#define PRECISION_c

#include "gemm_template_kernel_vbatched.dp.hpp"

#include "gemm_config/cgemm_param_nn.h"
#include "gemm_config/cgemm_param_nt.h"
#include "gemm_config/cgemm_param_tn.h"
#include "gemm_config/cgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_cgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    if(max_m <=0 || max_n <= 0 || max_k < 0) return;

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
                if(max_k < 64)
                {
                    if(max_k==8 && max_n==24)
                    gemm_template_vbatched_nn<magmaFloatComplex, version(NN,113), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    else if (max_n<32)
                    gemm_template_vbatched_nn<magmaFloatComplex, version(NN,124), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    else
                    gemm_template_vbatched_nn<magmaFloatComplex, version(NN,308), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_nn<magmaFloatComplex, version(NN,318), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 1: // nt
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_nt<magmaFloatComplex, version(NT,338), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_nt<magmaFloatComplex, version(NT,338), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_nt<magmaFloatComplex, version(NT,426), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 2: // nc
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_nt<magmaFloatComplex, version(NT,338), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_nt<magmaFloatComplex, version(NT,338), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_nt<magmaFloatComplex, version(NT,426), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 3: // tn
            {
                if(max_k < 16)
                {
                    gemm_template_vbatched_tn<magmaFloatComplex, version(TN,282), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_tn<magmaFloatComplex, version(TN,505), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 6: // cn
            {
                if(max_k < 16)
                {
                    gemm_template_vbatched_tn<magmaFloatComplex, version(TN,282), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_tn<magmaFloatComplex, version(TN,505), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 4: // tt
            {
                if(max_k < 16)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,73), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }else
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 5: // tc
            {
                if(max_k < 16)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,73), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }else
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 7: // ct
            {
                if(max_k < 16)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,73), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }else
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 8: // cc
            {
                if(max_k < 16)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,73), 1, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }else
                    {
                        gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 1, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<magmaFloatComplex, version(TT,175), 1, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        default:; // propose something
    }
}
