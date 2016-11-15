/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Chongxiao Cao
       @author Tingxing Dong
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"
#include "../control/batched_kernel_param.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"
#endif

#define h_A_tmp(i,j) (h_A_tmp + (i) + (j)*h_lda[k])

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm_vbatched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time=0, cpu_perf=0, cpu_time=0;
    double          error, lapack_error, magma_error, normalize, work[1];
    magma_int_t M, N, info;
    magma_int_t total_size_A_cpu, total_size_A_dev;
    magma_int_t total_size_B_cpu, total_size_B_dev;
    magma_int_t *Ak;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;
    magma_int_t max_M, max_N, max_Ak;
    
    magma_int_t *h_M, *h_N, *d_M, *d_N;
    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magma_int_t *h_ldb, *h_lddb, *d_lddb;
    magma_int_t *h_invA_size, *d_invA_size;

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    magmaDoubleComplex *h_A, *h_B, *h_Bcublas, *h_Bmagma, *h_Blapack, *h_X;
    magmaDoubleComplex *d_A, *d_B;
    
    magmaDoubleComplex **h_A_array = NULL;
    magmaDoubleComplex **h_B_array = NULL;
    magmaDoubleComplex **d_A_array = NULL;
    magmaDoubleComplex **d_B_array = NULL;
    
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dW4_displ  = NULL;
    
    magmaDoubleComplex **hinvA_array = NULL;
    magmaDoubleComplex **hwork_array = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **dwork_array = NULL;
    
     magmaDoubleComplex *h_A_tmp, *h_B_tmp;

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    int status = 0;
    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;
    
    TESTING_CHECK( magma_imalloc_cpu( &h_M, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_N, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &Ak,  batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_lda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_ldb, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_lddb, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_invA_size, batchCount) );
    
    TESTING_CHECK( magma_imalloc(&d_M, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_ldda, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_lddb, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_invA_size, batchCount) );
    
    double *Anorm, *Bnorm, *Cnorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Bnorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Cnorm, batchCount ));
    
    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_B_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hinvA_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hwork_array, batchCount*sizeof(magmaDoubleComplex*)) );
    
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_B_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&dinvA_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&dwork_array, batchCount*sizeof(magmaDoubleComplex*)) );
            
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s\n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    
    printf("%%              max   max\n");
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   LAPACK error\n");
    printf("%%============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            // guarantee reproducible sizes
            srand(1000);
            
            M = opts.msize[itest];
            N = opts.nsize[itest];
            total_size_A_cpu = 0;
            total_size_B_cpu = 0;
            total_size_A_dev = 0;
            total_size_B_dev = 0;
            gflops = 0;
            max_M = 0;
            max_N = 0;
            magma_int_t dinvA_batchSize = 0;
            magma_int_t dwork_batchSize = 0;
            for(int k = 0; k < batchCount; k++)
            {
                h_M[k] = 1 + (rand() % M);
                h_N[k] = 1 + (rand() % N);
                if(h_M[k] > max_M) max_M = h_M[k];
                if(h_N[k] > max_N) max_N = h_N[k];
                
                if ( opts.side == MagmaLeft ){
                    h_lda[k] = h_M[k];
                    Ak[k]  = h_M[k];
                }
                else{
                    h_lda[k] = h_N[k];
                    Ak[k]  = h_N[k];
                }
                h_ldb[k] = h_M[k];
                
                h_ldda[k] = magma_roundup( h_lda[k], opts.align );  // multiple of 32 by default
                h_lddb[k] = magma_roundup( h_ldb[k], opts.align );  // multiple of 32 by default
                
                total_size_A_cpu += Ak[k] * h_lda[k];
                total_size_A_dev += Ak[k] * h_ldda[k];
                
                total_size_B_cpu += h_N[k] * h_ldb[k];
                total_size_B_dev += h_N[k] * h_lddb[k];
                
                gflops += FLOPS_ZTRSM(opts.side, h_M[k], h_N[k]) / 1e9;
                h_invA_size[k] = magma_roundup( Ak[k], ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
                dinvA_batchSize += h_invA_size[k];
                dwork_batchSize += h_lddb[k]*h_N[k];
            }
            max_Ak = ( opts.side == MagmaLeft ) ? max_M : max_N;
            
            TESTING_CHECK( magma_zmalloc_cpu(&h_A,  total_size_A_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_B,  total_size_B_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_X,  total_size_B_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Blapack,  total_size_B_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Bcublas,  total_size_B_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Bmagma,  total_size_B_cpu) );
            TESTING_CHECK( magma_imalloc_cpu(&ipiv,  max_Ak) );
            
            TESTING_CHECK( magma_zmalloc(&d_A, total_size_A_dev) );
            TESTING_CHECK( magma_zmalloc(&d_B, total_size_B_dev) );
            
            TESTING_CHECK( magma_malloc( (void**) &dW1_displ,   batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dW2_displ,   batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dW3_displ,   batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dW4_displ,   batchCount * sizeof(magmaDoubleComplex*) ));
            
            magmaDoubleComplex* dinvA=NULL;
            magmaDoubleComplex* dwork=NULL; // invA and work are workspace in ztrsm
            
            TESTING_CHECK( magma_zmalloc( &dinvA, dinvA_batchSize ));
            TESTING_CHECK( magma_zmalloc( &dwork, dwork_batchSize ));
            
            h_A_array[0] = d_A;
            h_B_array[0] = d_B;
            hwork_array[0] = dwork;
            hinvA_array[0] = dinvA;
            for(int k = 1; k < batchCount; k++)
            {
                h_A_array[k] = h_A_array[k-1] + Ak[k-1] * h_ldda[k-1];
                h_B_array[k] = h_B_array[k-1] + h_N[k-1] * h_lddb[k-1];
                hinvA_array[k] = hinvA_array[k-1] + h_invA_size[k-1];
                hwork_array[k] = hwork_array[k-1] + h_lddb[k-1]*h_N[k-1];
            }
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_A_array, 1, d_A_array, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_B_array, 1, d_B_array, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), hinvA_array, 1, dinvA_array, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), hwork_array, 1, dwork_array, 1, opts.queue);
            
            memset(h_Bmagma, 0, total_size_B_cpu*sizeof(magmaDoubleComplex));
            for(int k = 0; k < batchCount; k++)
            {
                magmablas_zlaset( MagmaFull, h_lddb[k], h_N[k], c_zero, c_zero, hwork_array[k], h_lddb[k], opts.queue);
            }
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_zlarnv( &ione, ISEED, &total_size_A_cpu, h_A );
            h_A_tmp = h_A;
            for (int k=0; k < batchCount; k++)
            {
                lapackf77_zgetrf( &Ak[k], &Ak[k], h_A_tmp, &h_lda[k], ipiv, &info );
                for( int j = 0; j < Ak[k]; ++j )
                {
                    for( int i = 0; i < j; ++i ) {
                        *h_A_tmp(i,j) = *h_A_tmp(j,i);
                    }
                }
                h_A_tmp += Ak[k] * h_lda[k];
            }
            lapackf77_zlarnv( &ione, ISEED, &total_size_B_cpu, h_B );
            memcpy( h_Blapack, h_B, total_size_B_cpu*sizeof(magmaDoubleComplex) );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            h_A_tmp = h_A;
            h_B_tmp = h_B;
            for(int k = 0; k < batchCount; k++)
            {
                magma_zsetmatrix( Ak[k], Ak[k], h_A_tmp, h_lda[k], h_A_array[k], h_ldda[k], opts.queue);
                magma_zsetmatrix( h_M[k],  h_N[k], h_B_tmp, h_ldb[k], h_B_array[k], h_lddb[k], opts.queue);
                h_A_tmp += Ak[k] * h_lda[k];
                h_B_tmp += h_N[k] * h_ldb[k];
            }
            magma_setvector(batchCount, sizeof(magma_int_t), h_M, 1, d_M, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_lddb, 1, d_lddb, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_invA_size, 1, d_invA_size, 1, opts.queue);
            //////////////////////////////////////////////////////////
            
            magma_time = magma_sync_wtime( opts.queue );
            if (opts.version == 1) {
                magmablas_ztrsm_outofplace_vbatched(
                    opts.side, opts.uplo, opts.transA, opts.diag, 1,
                    d_M, d_N, alpha,
                    d_A_array,    d_ldda, // dA
                    d_B_array,    d_lddb, // dB
                    dwork_array,  d_lddb, // dX output
                    dinvA_array,  d_invA_size,
                    dW1_displ,   dW2_displ,
                    dW3_displ,   dW4_displ,
                    1, batchCount,
                    max_M, max_N, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                h_B_tmp = h_Bmagma;
                magmaDoubleComplex *work_tmp = dwork;
                for(int k = 0; k < batchCount; k++)
                {
                    magma_zgetmatrix( h_M[k], h_N[k], work_tmp, h_lddb[k], h_B_tmp, h_ldb[k], opts.queue);
                    h_B_tmp  += h_N[k] * h_ldb[k];
                    work_tmp += h_N[k] * h_lddb[k];
                }
            }
            else {
                magmablas_ztrsm_vbatched(
                    opts.side, opts.uplo, opts.transA, opts.diag,
                    d_M, d_N, alpha,
                    d_A_array, d_ldda,
                    d_B_array, d_lddb,
                    batchCount, opts.queue );
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                h_B_tmp = h_Bmagma;
                magmaDoubleComplex *d_B_tmp = d_B;
                for(int k = 0; k < batchCount; k++)
                {
                    magma_zgetmatrix( h_M[k], h_N[k], d_B_tmp, h_lddb[k], h_B_tmp, h_ldb[k], opts.queue);
                    h_B_tmp  += h_N[k] * h_ldb[k];
                    d_B_tmp  += h_N[k] * h_lddb[k];
                }
            }
             
            if ( opts.lapack )
            {
                /* =====================================================================
                   Performs operation using CPU BLAS
                   =================================================================== */
                // displace pointers for the cpu, reuse h_A_array, h_B_array
                h_A_array[0] = h_A;
                h_B_array[0] = h_Blapack;
                for(int k = 1; k < batchCount; k++){
                    h_A_array[k] = h_A_array[k-1] + Ak[k-1] * h_lda[k-1];
                    h_B_array[k] = h_B_array[k-1] + h_N[k-1] * h_ldb[k-1];
                }
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int s=0; s < batchCount; s++) {
                    blasf77_ztrsm(
                        lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &h_M[s], &h_N[s], &alpha,
                        h_A_array[s], &h_lda[s],
                        h_B_array[s], &h_ldb[s] );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            
                /* =====================================================================
                   Check the result
                   =================================================================== */
                // ||b - 1/alpha*A*x|| / (||A||*||x||)
                magmaDoubleComplex inv_alpha = MAGMA_Z_DIV( c_one, alpha );
                double normR, normX, normA;
                magma_error  = 0;
                memcpy( h_X, h_Bmagma, total_size_B_cpu*sizeof(magmaDoubleComplex) );
                magmaDoubleComplex *h_X_tmp = h_X;
                magmaDoubleComplex *h_Bmagma_tmp = h_Bmagma;
                h_B_tmp = h_B;
                // check magma
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t NN = h_ldb[s]*h_N[s];
                    normA = lapackf77_zlantr( "M",
                                              lapack_uplo_const(opts.uplo),
                                              lapack_diag_const(opts.diag),
                                              &Ak[s], &Ak[s], h_A_array[s], &h_lda[s], work );
                    blasf77_ztrmm(
                        lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &h_M[s], &h_N[s], &inv_alpha,
                        h_A_array[s], &h_lda[s],
                        h_X_tmp,      &h_ldb[s] );
                    blasf77_zaxpy( &NN, &c_neg_one, h_B_tmp, &ione, h_X_tmp, &ione );
    
                    normR = lapackf77_zlange( "M", &h_M[s], &h_N[s], h_X_tmp,      &h_ldb[s], work );
                    normX = lapackf77_zlange( "M", &h_M[s], &h_N[s], h_Bmagma_tmp, &h_ldb[s], work );
                    normalize = normX*normA;
                    if (normalize == 0)
                        normalize = 1;
                    error = normR / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                    
                    h_X_tmp += h_N[s] * h_ldb[s];
                    h_B_tmp += h_N[s] * h_ldb[s];
                    h_Bmagma_tmp += h_N[s] * h_ldb[s];
                }
                bool okay = (magma_error < tol);
                status += ! okay;
    
                // check lapack
                // this verifies that the matrix wasn't so bad that it couldn't be solved accurately.
                magmaDoubleComplex *h_Blapack_tmp = h_Blapack;
                h_B_tmp = h_B;
                h_X_tmp = h_X;
                
                lapack_error = 0;
                memcpy( h_X, h_Blapack, total_size_B_cpu*sizeof(magmaDoubleComplex) );
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t NN = h_ldb[s]*h_N[s];
                    normA = lapackf77_zlantr( "M",
                                              lapack_uplo_const(opts.uplo),
                                              lapack_diag_const(opts.diag),
                                              &Ak[s], &Ak[s], h_A_array[s], &h_lda[s], work );
                    blasf77_ztrmm(
                        lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &h_M[s], &h_N[s], &inv_alpha,
                        h_A_array[s], &h_lda[s],
                        h_X_tmp,  &h_ldb[s] );
    
                    blasf77_zaxpy( &NN, &c_neg_one, h_B_tmp, &ione, h_X_tmp, &ione );
                    normR = lapackf77_zlange( "M", &h_M[s], &h_N[s], h_X_tmp,       &h_ldb[s], work );
                    normX = lapackf77_zlange( "M", &h_M[s], &h_N[s], h_Blapack_tmp, &h_ldb[s], work );
                    normalize = normX*normA;
                    if (normalize == 0)
                        normalize = 1;
                    error = normR / normalize;
                    lapack_error = magma_max_nan( error, lapack_error );
                    
                    h_X_tmp += h_N[s] * h_ldb[s];
                    h_B_tmp += h_N[s] * h_ldb[s];
                    h_Blapack_tmp += h_N[s] * h_ldb[s];
                }
                
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e      %8.2e   %s\n",
                        (long long)batchCount, (long long)max_M, (long long)max_N,
                        magma_perf,  1000.*magma_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, lapack_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---           ---\n",
                        (long long)batchCount, (long long)max_M, (long long)max_N,
                        magma_perf,  1000.*magma_time);
            }
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( h_Blapack );
            magma_free_cpu( h_Bcublas );
            magma_free_cpu( h_Bmagma  );
            magma_free_cpu( ipiv );
            
            magma_free( d_A );
            magma_free( d_B );
            magma_free( dW1_displ );
            magma_free( dW2_displ );
            magma_free( dW3_displ );
            magma_free( dW4_displ );
            magma_free( dinvA );
            magma_free( dwork );
            
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( Ak );
    magma_free_cpu( h_lda );
    magma_free_cpu( h_ldb );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_lddb );

    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );
    magma_free_cpu( Cnorm );

    magma_free_cpu( h_A_array);
    magma_free_cpu( h_B_array);
    magma_free_cpu( h_invA_size );
    magma_free_cpu( hinvA_array);
    magma_free_cpu( hwork_array);
    
    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_lddb );
    magma_free( d_A_array );
    magma_free( d_B_array );
    magma_free( dinvA_array );
    magma_free( dwork_array );
    magma_free( d_invA_size );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
