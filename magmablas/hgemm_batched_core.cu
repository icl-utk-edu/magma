/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_h
#include "hgemm_template_kernel_batched.cuh"
#include "./gemm_config/hgemm_param.h"
#define version(v) NN_V_ ## v

extern "C" magma_int_t 
magmablas_hgemm_batched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    magmaHalf alpha,
    magmaHalf const * const * dAarray, magma_int_t ldda,
    magmaHalf const * const * dBarray, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dCarray, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magma_int_t arch = magma_getdevice_arch();
    if(arch < 700) {
        printf("%s: architecture %lld is not supported\n", __func__, (long long)arch);
        return -14;
    }

#if CUDA_VERSION >= 9000
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

    switch(shape){
        case 0:    // nn
        {
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_nn<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nn<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nn<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nn<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nn<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nn<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nn<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nn<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nn<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_nn<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nn<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nn<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nn<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nn<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nn<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nn<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nn<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        case 1:    // nt 
        case 2:    // nc
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_nt<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nt<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nt<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nt<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nt<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nt<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nt<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nt<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nt<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_nt<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nt<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nt<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nt<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nt<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nt<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nt<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nt<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        case 3:    // tn
        case 6:    // cn
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_tn<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tn<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tn<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tn<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tn<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tn<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tn<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tn<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tn<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_tn<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tn<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tn<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tn<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tn<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tn<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tn<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tn<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        case 4:    // tt
        case 5:    // tc
        case 7:    // ct
        case 8:    // cc
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_tt<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tt<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tt<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tt<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tt<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tt<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tt<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tt<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tt<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_tt<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tt<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tt<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tt<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tt<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tt<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tt<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tt<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        default:; // propose something
    }
#endif
    return 0;
}
