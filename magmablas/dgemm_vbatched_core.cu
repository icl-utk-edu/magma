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
#define PRECISION_d
#include "gemm_template_kernel_vbatched.cuh"

#include "gemm_config/dgemm_param_nn.h"
#include "gemm_config/dgemm_param_nt.h"
#include "gemm_config/dgemm_param_tn.h"
#include "gemm_config/dgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_dgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    double beta,
    double              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    if (max_m <= 0 || max_n <= 0 || max_k < 0) return;

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
                if(max_k < 32)
                {
                    if(max_k==8 && max_n==24)
                    gemm_template_vbatched_nn<double, version(NN,32), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    else if (max_n<32)
                    gemm_template_vbatched_nn<double, version(NN,49), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    else
                    gemm_template_vbatched_nn<double, version(NN,111), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 80)
                    {
                        gemm_template_vbatched_nn<double, version(NN,93), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_nn<double, version(NN,111), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 1: // nt
            {
                if(max_k < 128)
                {
                    gemm_template_vbatched_nt<double, version(NT,160), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_nt<double, version(NT,160), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_nt<double, version(NT,190), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 2: // nc
            {
                if(max_k < 128)
                {
                    gemm_template_vbatched_nt<double, version(NT,160), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_nt<double, version(NT,160), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_nt<double, version(NT,190), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 3: // tn
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_tn<double, version(TN,207), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_tn<double, version(TN,207), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tn<double, version(TN,209), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 6: // cn
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_tn<double, version(TN,207), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_tn<double, version(TN,207), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tn<double, version(TN,209), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 4: // tt
            {
                if(max_k < 128)
                {
                    gemm_template_vbatched_tt<double, version(TT,81), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_tt<double, version(TT,81), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<double, version(TT,85), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 5: // tc
            {
                if(max_k < 128)
                {
                    gemm_template_vbatched_tt<double, version(TT,81), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_tt<double, version(TT,81), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<double, version(TT,85), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 7: // ct
            {
                if(max_k < 128)
                {
                    gemm_template_vbatched_tt<double, version(TT,81), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_tt<double, version(TT,81), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<double, version(TT,85), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        case 8: // cc
            {
                if(max_k < 128)
                {
                    gemm_template_vbatched_tt<double, version(TT,81), 1, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    if(max_m < 256)
                    {
                        gemm_template_vbatched_tt<double, version(TT,81), 1, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<double, version(TT,85), 1, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
            }
            break;
        default:; // propose something
    }
}
