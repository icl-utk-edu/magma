/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/
#include "magma_internal.h"

#define CHECKER_TX    (128)

/******************************************************************************/
// GEMM checker
// ------------
__global__ void 
gemm_vbatched_checker( 
        magma_trans_t transA, magma_trans_t transB, 
        magma_int_t* m, magma_int_t* n, magma_int_t* k,
        magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_m = (int)m[gtx];
    const int local_n = (int)n[gtx];
    const int local_k = (int)k[gtx];
    
    const int local_ldda = (int)ldda[gtx];
    const int local_lddb = (int)lddb[gtx];
    const int local_lddc = (int)lddc[gtx];
    
    if( local_m < 0 ) m[batchCount] = -1;
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_k < 0 ) k[batchCount] = -1;
    if( transA == MagmaNoTrans ? local_ldda < local_m : local_ldda < local_k ) ldda[batchCount] = -1;
    if( transB == MagmaNoTrans ? local_lddb < local_k : local_lddb < local_n ) lddb[batchCount] = -1;
    if( local_lddc < local_m ) lddc[batchCount] = -1;
}


/******************************************************************************/
// driver
extern "C" magma_int_t 
magma_gemm_vbatched_checker(
        magma_trans_t transA, magma_trans_t transB, 
        magma_int_t* m, magma_int_t* n, magma_int_t* k, 
        magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc,  
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t m_err = 0, ldda_err = 0;
    magma_int_t n_err = 0, lddb_err = 0;
    magma_int_t k_err = 0, lddc_err = 0;
    
    // Assume no error
    magma_isetvector_async(1, &m_err   , 1, &m[batchCount]   , 1, queue);
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &k_err   , 1, &k[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddb_err, 1, &lddb[batchCount], 1, queue);
    magma_isetvector_async(1, &lddc_err, 1, &lddc[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    gemm_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(transA, transB, m, n, k, ldda, lddb, lddc, batchCount);
    
    magma_igetvector_async(1, &m[batchCount]   , 1, &m_err   , 1, queue);
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &k[batchCount]   , 1, &k_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddb[batchCount], 1, &lddb_err, 1, queue);
    magma_igetvector_async(1, &lddc[batchCount], 1, &lddc_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m_err < 0 )
        info = -3;
    else if ( n_err < 0 )
        info = -4;
    else if ( k_err < 0 )
        info = -5;
    else if ( ldda_err < 0 )
        info = -8;
    else if ( lddb_err < 0 )
        info = -10;
    else if ( lddc_err < 0 )
        info = -13;
    else if ( batchCount < 0)
        info = -14;
    
    return info;
}


/******************************************************************************/
// TRSM checker
// ------------
__global__ void 
trsm_vbatched_checker( 
        magma_side_t side, magma_int_t* m, magma_int_t* n,
        magma_int_t* ldda, magma_int_t* lddb, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_m = (int)m[gtx];
    const int local_n = (int)n[gtx];
    const int local_ldda = (int)ldda[gtx];
    const int local_lddb = (int)lddb[gtx];
    
    const int nrowA = (side == MagmaLeft ? local_m : local_n);
    
    if( local_m < 0 ) m[batchCount] = -1; 
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_ldda < max(1, nrowA) ) ldda[batchCount] = -1;
    if( local_lddb < max(1, local_m) ) lddb[batchCount] = -1;
}


/******************************************************************************/
// driver 
extern "C" magma_int_t 
magma_trsm_vbatched_checker( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n,
        magma_int_t* ldda, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    magma_int_t m_err = 0, ldda_err = 0;
    magma_int_t n_err = 0, lddb_err = 0;
    
    // Assume no error
    magma_isetvector_async(1, &m_err   , 1, &m[batchCount]   , 1, queue);
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddb_err, 1, &lddb[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    trsm_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(side, m, n, ldda, lddb, batchCount);
    
    magma_igetvector_async(1, &m[batchCount]   , 1, &m_err   , 1, queue);
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddb[batchCount], 1, &lddb_err, 1, queue);
    magma_queue_sync( queue );
    
    if ( side != MagmaLeft && side != MagmaRight ) 
        info = -1;
    else if ( uplo != MagmaUpper && uplo != MagmaLower ) 
        info = -2;
    else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) 
        info = -3;
    else if ( diag != MagmaUnit && diag != MagmaNonUnit ) 
        info = -4;
    else if ( m_err < 0 ) 
        info = -5;
    else if ( n_err < 0 ) 
        info = -6;
    else if ( ldda_err < 0 )
        info = -9;
    else if ( lddb_err < 0 )
        info = -11;
    else if ( batchCount < 0 )
        info = -12;

    return info;
}


/******************************************************************************/
// SYRK/HERK checker
// ------------
__global__ void 
herk_vbatched_checker(
        magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddc, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_n = (int)n[gtx];
    const int local_k = (int)k[gtx];
    const int local_ldda = (int)ldda[gtx];
    const int local_lddc = (int)lddc[gtx];
    
    const int nrowA = ( trans == MagmaNoTrans ? local_n : local_k);
    
    if( local_n < 0 ) n[batchCount] = -1; 
    if( local_k < 0 ) k[batchCount] = -1;
    if( local_ldda < nrowA  ) ldda[batchCount] = -1;
    if( local_lddc < local_n) lddc[batchCount] = -1;
}


/******************************************************************************/
// driver - ssyrk, dsyrk, csyrk, zsyrk
extern "C" magma_int_t 
magma_syrk_vbatched_checker(
        magma_int_t icomplex, 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    magma_int_t n_err = 0, ldda_err = 0;
    magma_int_t k_err = 0, lddc_err = 0;
    
    // Assume no error
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &k_err   , 1, &k[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddc_err, 1, &lddc[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    herk_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(trans, n, k, ldda, lddc, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &k[batchCount]   , 1, &k_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddc[batchCount], 1, &lddc_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( ( icomplex == 0 && (trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans) ) || 
              ( icomplex == 1 && (trans != MagmaNoTrans && trans != MagmaTrans) )
            )
        info = -2;
    else if ( n_err < 0 )
        info = -3;
    else if ( k_err < 0 )
        info = -4;
    else if (ldda_err < 0 )  
        info = -7;
    else if ( lddc_err < 0 ) 
        info = -10;
    else if ( batchCount < 0 )
        info = -11;

    return info;    
}


/******************************************************************************/
// driver - cherk, zherk
extern "C" magma_int_t 
magma_herk_vbatched_checker( 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    magma_int_t n_err = 0, ldda_err = 0;
    magma_int_t k_err = 0, lddc_err = 0;
    
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &k_err   , 1, &k[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddc_err, 1, &lddc[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    herk_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(trans, n, k, ldda, lddc, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &k[batchCount]   , 1, &k_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddc[batchCount], 1, &lddc_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
        info = -2;
    else if ( n_err < 0 )
        info = -3;
    else if ( k_err < 0 )
        info = -4;
    else if (ldda_err < 0 )  
        info = -7;
    else if ( lddc_err < 0 ) 
        info = -10;
    else if ( batchCount < 0 )
        info = -11;

    return info;    
}


/******************************************************************************/
// SYR2K/HER2K checker
// ------------
__global__ void 
her2k_vbatched_checker(
        magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_n = (int)n[gtx];
    const int local_k = (int)k[gtx];
    const int local_ldda = (int)ldda[gtx];
    const int local_lddb = (int)lddb[gtx];
    const int local_lddc = (int)lddc[gtx];
    
    const int nrowAB = ( trans == MagmaNoTrans ? local_n : local_k);
    
    if( local_n < 0 ) n[batchCount] = -1; 
    if( local_k < 0 ) k[batchCount] = -1;
    if( local_ldda < max(1, nrowAB ) ) ldda[batchCount] = -1;
    if( local_lddb < max(1, nrowAB ) ) lddb[batchCount] = -1;
    if( local_lddc < max(1, local_n) ) lddc[batchCount] = -1;
}


/******************************************************************************/
// driver - ssyr2k, dsyr2k, csyr2k, zsyr2k
extern "C" magma_int_t 
magma_syr2k_vbatched_checker(
        magma_int_t icomplex, 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    magma_int_t n_err = 0, k_err = 0;
    magma_int_t ldda_err = 0, lddb_err = 0, lddc_err = 0;
    
    // Assume no error
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &k_err   , 1, &k[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddb_err, 1, &lddb[batchCount], 1, queue);
    magma_isetvector_async(1, &lddc_err, 1, &lddc[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    her2k_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(trans, n, k, ldda, lddb, lddc, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &k[batchCount]   , 1, &k_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddb[batchCount], 1, &lddb_err, 1, queue);
    magma_igetvector_async(1, &lddc[batchCount], 1, &lddc_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( ( icomplex == 0 && (trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans) ) || 
              ( icomplex == 1 && (trans != MagmaNoTrans && trans != MagmaTrans                           ) )
            )
        info = -2;
    else if ( n_err < 0 )
        info = -3;
    else if ( k_err < 0 )
        info = -4;
    else if (ldda_err < 0 )  
        info = -7;
    else if (lddb_err < 0 )  
        info = -9;
    else if ( lddc_err < 0 ) 
        info = -12;
    else if ( batchCount < 0 )
        info = -13;

    return info;    
}


/******************************************************************************/
// driver - cher2k, zher2k
extern "C" magma_int_t 
magma_her2k_vbatched_checker( 
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t *n, magma_int_t *k, 
        magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    
    magma_int_t n_err = 0, k_err = 0;
    magma_int_t ldda_err = 0, lddb_err = 0, lddc_err = 0;
    
    // assume no error
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &k_err   , 1, &k[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddb_err, 1, &lddb[batchCount], 1, queue);
    magma_isetvector_async(1, &lddc_err, 1, &lddc[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    her2k_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(trans, n, k, ldda, lddb, lddc, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &k[batchCount]   , 1, &k_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddb[batchCount], 1, &lddb_err, 1, queue);
    magma_igetvector_async(1, &lddc[batchCount], 1, &lddc_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
        info = -2;
    else if ( n_err < 0 )
        info = -3;
    else if ( k_err < 0 )
        info = -4;
    else if (ldda_err < 0 )  
        info = -7;
    else if (lddb_err < 0 )  
        info = -9;
    else if ( lddc_err < 0 ) 
        info = -12;
    else if ( batchCount < 0 )
        info = -13;

    return info;    
}


/******************************************************************************/
// HEMM checker
// ------------
__global__ void 
hemm_vbatched_checker( 
        magma_side_t side, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_m = (int)m[gtx];
    const int local_n = (int)n[gtx];
    const int local_nrowa = (side == MagmaLeft ? local_m : local_n);
    
    const int local_ldda = (int)ldda[gtx];
    const int local_lddb = (int)lddb[gtx];
    const int local_lddc = (int)lddc[gtx];
    
    if( local_m < 0 ) m[batchCount] = -1;
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_ldda < max(1,local_nrowa) ) ldda[batchCount] = -1;
    if( local_lddb < max(1,local_m)     ) lddb[batchCount] = -1;
    if( local_lddc < max(1,local_m)     ) lddc[batchCount] = -1;
}


/******************************************************************************/
// driver - HEMM
extern "C" magma_int_t 
magma_hemm_vbatched_checker(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc,  
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t m_err = 0, n_err = 0;
    magma_int_t ldda_err = 0, lddb_err = 0, lddc_err = 0;
    
    // Assume no error
    magma_isetvector_async(1, &m_err   , 1, &m[batchCount]   , 1, queue);
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddb_err, 1, &lddb[batchCount], 1, queue);
    magma_isetvector_async(1, &lddc_err, 1, &lddc[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    hemm_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(side, m, n, ldda, lddb, lddc, batchCount);
    
    magma_igetvector_async(1, &m[batchCount]   , 1, &m_err   , 1, queue);
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddb[batchCount], 1, &lddb_err, 1, queue);
    magma_igetvector_async(1, &lddc[batchCount], 1, &lddc_err, 1, queue);
    magma_queue_sync( queue );
    
    if ( side != MagmaLeft && side != MagmaRight )
        info = -1;
    else if (uplo != MagmaLower && uplo != MagmaUpper )
        info = -2;
    else if ( m_err < 0 )
        info = -3;
    else if ( n_err < 0 )
        info = -4;
    else if ( ldda_err < 0 )
        info = -7;
    else if ( lddb_err < 0 )
        info = -9;
    else if ( lddc_err < 0 )
        info = -12;
    else if ( batchCount < 0)
        info = -13;
    
    return info;
}


/******************************************************************************/
// GEMV checker
// ------------
__global__ void 
gemv_vbatched_checker( 
        magma_trans_t trans, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_m = (int)m[gtx];
    const int local_n = (int)n[gtx];
    
    const int local_ldda = (int)ldda[gtx];
    const int local_incx = (int)incx[gtx];
    const int local_incy = (int)incy[gtx];
    
    if( local_m < 0 ) m[batchCount] = -1;
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_ldda < local_m ) ldda[batchCount] = -1;
    if( local_incx == 0 ) incx[batchCount] = -1;
    if( local_incy == 0 ) incy[batchCount] = -1;
}


/******************************************************************************/
// driver
extern "C" magma_int_t 
magma_gemv_vbatched_checker(
        magma_trans_t trans, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy,  
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t m_err = 0, n_err = 0; 
    magma_int_t ldda_err = 0; 
    magma_int_t incx_err = 0, incy_err = 0;
    
    // assume no error
    magma_isetvector_async(1, &m_err   , 1, &m[batchCount]   , 1, queue);
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &incx_err, 1, &incx[batchCount], 1, queue);
    magma_isetvector_async(1, &incy_err, 1, &incy[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    gemv_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(trans , m, n, ldda, incx, incy, batchCount);
    
    magma_igetvector_async(1, &m[batchCount]   , 1, &m_err   , 1, queue);
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &incx[batchCount], 1, &incx_err, 1, queue);
    magma_igetvector_async(1, &incy[batchCount], 1, &incy_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m_err < 0 )
        info = -2;
    else if ( n_err < 0 )
        info = -3;
    else if ( ldda_err < 0 )
        info = -6;
    else if ( incx_err < 0 )
        info = -8;
    else if ( incy_err < 0 )
        info = -11;
    else if ( batchCount < 0)
        info = -12;

    return info;
}
/******************************************************************************/
// HEMV checker
// ------------
__global__ void 
hemv_vbatched_checker( 
        magma_uplo_t uplo, 
        magma_int_t* n, magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_n = (int)n[gtx];
    const int local_ldda = (int)ldda[gtx];
    const int local_incx = (int)incx[gtx];
    const int local_incy = (int)incy[gtx];
    
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_ldda < local_n ) ldda[batchCount] = -1;
    if( local_incx == 0 ) incx[batchCount] = -1;
    if( local_incy == 0 ) incy[batchCount] = -1;
}


/******************************************************************************/
// driver
extern "C" magma_int_t 
magma_hemv_vbatched_checker(
        magma_uplo_t uplo, 
        magma_int_t* n, magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy,  
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t n_err = 0; 
    magma_int_t ldda_err = 0; 
    magma_int_t incx_err = 0, incy_err = 0;
    
    // assume no error
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &incx_err, 1, &incx[batchCount], 1, queue);
    magma_isetvector_async(1, &incy_err, 1, &incy[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    hemv_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(uplo , n, ldda, incx, incy, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &incx[batchCount], 1, &incx_err, 1, queue);
    magma_igetvector_async(1, &incy[batchCount], 1, &incy_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( n_err < 0 )
        info = -2;
    else if ( ldda_err < 0 )
        info = -5;
    else if ( incx_err < 0 )
        info = -7;
    else if ( incy_err < 0 )
        info = -10;
    else if ( batchCount < 0)
        info = -11;

    return info;
}


/******************************************************************************/
// AXPY checker
// ------------
__global__ void 
axpy_vbatched_checker(
        magma_int_t *n, 
        magma_int_t *incx, magma_int_t *incy, 
        int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_n    = (int)n[gtx];
    const int local_incx = (int)incx[gtx];
    const int local_incy = (int)incy[gtx];
    
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_incx == 0 ) incx[batchCount] = -1; 
    if( local_incy == 0 ) incy[batchCount] = -1; 
}


/******************************************************************************/
// driver
extern "C" magma_int_t 
magma_axpy_vbatched_checker( 
        magma_int_t *n, 
        magma_int_t *incx, magma_int_t *incy, 
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    magma_int_t n_err = 0, incx_err = 0, incy_err = 0; 
        
    // assume no error
    magma_isetvector_async(1, &n_err    , 1, &n[batchCount]    , 1, queue);
    magma_isetvector_async(1, &incx_err , 1, &incx[batchCount] , 1, queue);
    magma_isetvector_async(1, &incy_err , 1, &incy[batchCount] , 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    axpy_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(n, incx, incy, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]    , 1, &n_err    , 1, queue);
    magma_igetvector_async(1, &incx[batchCount] , 1, &incx_err , 1, queue);
    magma_igetvector_async(1, &incy[batchCount] , 1, &incy_err , 1, queue);
    magma_queue_sync( queue );
    
    if      ( n_err < 0 )
        info = -1;
    else if ( incx_err < 0 )
        info = -4;
    else if ( incy_err < 0 )
        info = -6;
    else if ( batchCount < 0 )
        info = -7;
    
    return info;
}


/******************************************************************************/
// TRMM checker
// ------------
__global__ void 
trmm_vbatched_checker( 
        magma_side_t side,  
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* lddb, 
        magma_int_t batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_m = (int)m[gtx];
    const int local_n = (int)n[gtx];
    const int local_ldda = (int)ldda[gtx];
    const int local_lddb = (int)lddb[gtx];
    const int nrowA = ( side == MagmaLeft ? local_m : local_n );
    
    if( local_m < 0 ) m[batchCount] = -1;
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_ldda < max(1, nrowA) ) ldda[batchCount] = -1;
    if( local_lddb < max(1, local_m) ) lddb[batchCount] = -1;
}


/******************************************************************************/
// driver
extern "C" magma_int_t 
magma_trmm_vbatched_checker(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magma_int_t* ldda, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t m_err = 0, ldda_err = 0;
    magma_int_t n_err = 0, lddb_err = 0;
    
    // Assume no error
    magma_isetvector_async(1, &m_err   , 1, &m[batchCount]   , 1, queue);
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    magma_isetvector_async(1, &lddb_err, 1, &lddb[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    trmm_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(side, m, n, ldda, lddb, batchCount);
    
    magma_igetvector_async(1, &m[batchCount]   , 1, &m_err   , 1, queue);
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_igetvector_async(1, &lddb[batchCount], 1, &lddb_err, 1, queue);
    magma_queue_sync( queue );
    
    
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m_err < 0) {
        info = -5;
    } else if (n_err < 0) {
        info = -6;
    } else if (ldda_err < 0) {
        info = -9;
    } else if (lddb_err < 0) {
        info = -11;
    } else if (batchCount < 0) {
        info = -12;
    }
    
    return info;
}
/******************************************************************************/
// POTRF checker
// ------------
__global__ void 
potrf_vbatched_checker( 
        magma_uplo_t uplo, 
        magma_int_t* n, magma_int_t* ldda, int batchCount)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * CHECKER_TX + tx; 
    if(gtx >= batchCount) return;
    
    const int local_n = (int)n[gtx];
    const int local_ldda = (int)ldda[gtx];
    
    if( local_n < 0 ) n[batchCount] = -1;
    if( local_ldda < local_n ) ldda[batchCount] = -1;
}


/******************************************************************************/
// driver
extern "C" magma_int_t 
magma_potrf_vbatched_checker(
        magma_uplo_t uplo, 
        magma_int_t* n, magma_int_t* ldda,  
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t n_err = 0; 
    magma_int_t ldda_err = 0; 
    
    // assume no error
    magma_isetvector_async(1, &n_err   , 1, &n[batchCount]   , 1, queue);
    magma_isetvector_async(1, &ldda_err, 1, &ldda[batchCount], 1, queue);
    
    // launch the checker kernel
    dim3 grid( magma_ceildiv(batchCount, CHECKER_TX), 1, 1 );
    dim3 threads( CHECKER_TX, 1, 1 );
    potrf_vbatched_checker<<< grid, threads, 0, queue->cuda_stream() >>>(uplo , n, ldda, batchCount);
    
    magma_igetvector_async(1, &n[batchCount]   , 1, &n_err   , 1, queue);
    magma_igetvector_async(1, &ldda[batchCount], 1, &ldda_err, 1, queue);
    magma_queue_sync( queue );
    
    if      ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( n_err < 0 )
        info = -2;
    else if ( ldda_err < 0 )
        info = -4;
    else if ( batchCount < 0)
        info = -6;

    return info;
}
