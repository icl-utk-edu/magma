/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah
       
       Implementation of batch BLAS on the host ( CPU ) using OpenMP
*/
#include "magma_internal.h"
#include "commonblas_z.h"

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

/*******************************************************************************/
extern "C" void
blas_zgemm_batched( 
        magma_trans_t transA, magma_trans_t transB, 
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDoubleComplex alpha,
        magmaDoubleComplex const * const * hA_array, magma_int_t lda,
        magmaDoubleComplex const * const * hB_array, magma_int_t ldb,
        magmaDoubleComplex beta,
        magmaDoubleComplex **hC_array, magma_int_t ldc, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_zgemm( lapack_trans_const(transA),
                       lapack_trans_const(transB),
                       &m, &n, &k,
                       &alpha, hA_array[i], &lda,
                               hB_array[i], &ldb, 
                       &beta,  hC_array[i], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_ztrsm_batched( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **hA_array, magma_int_t lda,
        magmaDoubleComplex **hB_array, magma_int_t ldb, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_ztrsm(
            lapack_side_const(side), lapack_uplo_const(uplo),
            lapack_trans_const(transA), lapack_diag_const(diag),
            &m, &n, &alpha,
            hA_array[s], &lda,
            hB_array[s], &ldb );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_ztrmm_batched( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **hA_array, magma_int_t lda,
        magmaDoubleComplex **hB_array, magma_int_t ldb, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_ztrmm(
            lapack_side_const(side), lapack_uplo_const(uplo),
            lapack_trans_const(transA), lapack_diag_const(diag),
            &m, &n, &alpha,
            hA_array[s], &lda,
            hB_array[s], &ldb );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_zhemm_batched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        magmaDoubleComplex alpha, 
        magmaDoubleComplex **hA_array, magma_int_t lda,
        magmaDoubleComplex **hB_array, magma_int_t ldb, 
        magmaDoubleComplex beta, 
        magmaDoubleComplex **hC_array, magma_int_t ldc, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_zhemm( lapack_side_const(side),
                       lapack_uplo_const(uplo),
                       &m, &n,
                       &alpha, hA_array[i], &lda,
                               hB_array[i], &ldb,
                       &beta,  hC_array[i], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    double alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda,
    double beta,  magmaDoubleComplex               **hC_array, magma_int_t ldc, 
    magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_zherk( lapack_uplo_const(uplo),
                       lapack_trans_const(trans),
                       &n, &k,
                       &alpha, hA_array[s], &lda,
                       &beta,  hC_array[s], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_zher2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda,
                              magmaDoubleComplex const * const * hB_array, magma_int_t ldb, 
    double beta,              magmaDoubleComplex               **hC_array, magma_int_t ldc, 
    magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_zher2k( lapack_uplo_const(uplo),
                        lapack_trans_const(trans),
                        &n, &k,
                        &alpha, hA_array[i], &lda,
                                hB_array[i], &ldb,
                        &beta,  hC_array[i], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

