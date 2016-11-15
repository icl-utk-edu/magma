/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Chongxiao Cao
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

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrmm_vbatched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          error, magma_error, normalize, work[1];
    magma_int_t M, N, max_M, max_N;
    magma_int_t total_sizeA_cpu, total_sizeB_cpu;
    magma_int_t total_sizeA_dev, total_sizeB_dev;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magmaDoubleComplex **hA_array, **hB_array;
    magmaDoubleComplex **dA_array, **dB_array;
    magmaDoubleComplex *h_A, *h_B, *h_Bmagma;
    magmaDoubleComplex_ptr d_A, d_B;
    magmaDoubleComplex *h_A_tmp, *h_B_tmp;
    magmaDoubleComplex *d_A_tmp, *d_B_tmp, *h_Bmagma_tmp;
    magma_int_t *h_M, *h_N, *h_lda, *h_ldb, *h_ldda, *h_lddb;
    magma_int_t *d_M, *d_N, *d_ldda, *d_lddb;
    magma_int_t *h_Ak;
    
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    int status = 0;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    magma_int_t batchCount = opts.batchcount;
    
    TESTING_CHECK( magma_imalloc_cpu( &h_M,    batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_N,    batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu( &h_lddb, batchCount) );
    
    TESTING_CHECK( magma_imalloc( &d_M,    (batchCount+1)) );
    TESTING_CHECK( magma_imalloc( &d_N,    (batchCount+1)) );
    TESTING_CHECK( magma_imalloc( &d_ldda, (batchCount+1)) );
    TESTING_CHECK( magma_imalloc( &d_lddb, (batchCount+1)) );
    
    double *Anorm, *Bnorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Bnorm, batchCount ));
    
    TESTING_CHECK( magma_malloc_cpu((void**)&hA_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hB_array, batchCount*sizeof(magmaDoubleComplex*)) );
    
    TESTING_CHECK( magma_malloc((void**)&dA_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&dB_array, batchCount*sizeof(magmaDoubleComplex*)) );
    
    // See testing_zgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s\n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%              max   max\n");
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            
            srand( 1000 );
            
            if ( opts.side == MagmaLeft ) {
                h_lda = h_M;
                h_Ak  = h_M;
            }
            else {
                h_lda = h_N;
                h_Ak  = h_N;
            }
            h_ldb = h_M;
            
            gflops = 0;
            max_M = max_N = 0;
            total_sizeA_cpu = total_sizeA_dev = 0;
            total_sizeB_cpu = total_sizeB_dev = 0;
            for (int i = 0; i < batchCount; i++) {
                h_M[i] = 1 + ( rand() % M );
                h_N[i] = 1 + ( rand() % N );
                
                h_ldda[i] = magma_roundup( h_lda[i], opts.align );  // multiple of 32 by default
                h_lddb[i] = magma_roundup( h_ldb[i], opts.align );  // multiple of 32 by default
                
                total_sizeA_cpu += h_lda[i]  * h_Ak[i];
                total_sizeB_cpu += h_ldb[i]  * h_N[i];
                total_sizeA_dev += h_ldda[i] * h_Ak[i];
                total_sizeB_dev += h_lddb[i] * h_N[i];
                
                max_M = max( max_M, h_M[i] );
                max_N = max( max_N, h_N[i] );
                
                gflops += FLOPS_ZTRMM(opts.side, h_M[i], h_N[i]);
            }
            gflops /= 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,      total_sizeA_cpu ) );
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,      total_sizeB_cpu ) );
            TESTING_CHECK( magma_zmalloc_cpu( &h_Bmagma, total_sizeB_cpu ) );
            
            TESTING_CHECK( magma_zmalloc( &d_A, total_sizeA_dev ) );
            TESTING_CHECK( magma_zmalloc( &d_B, total_sizeB_dev ) );
            
            // assign gpu pointers
            hA_array[0] = d_A;
            hB_array[0] = d_B;
            for (int i = 1; i < batchCount; i++) {
                hA_array[i] = hA_array[i-1] + h_ldda[i-1] * h_Ak[i-1];
                hB_array[i] = hB_array[i-1] + h_lddb[i-1] * h_N[i-1];
            }
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), hA_array, 1, dA_array, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), hB_array, 1, dB_array, 1, opts.queue);
            
            // send the sizes
            magma_setvector(batchCount, sizeof(magma_int_t), h_M, 1, d_M, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_lddb, 1, d_lddb, 1, opts.queue);
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &total_sizeA_cpu, h_A );
            lapackf77_zlarnv( &ione, ISEED, &total_sizeB_cpu, h_B );

            // Compute norms for error
            h_A_tmp = h_A;
            h_B_tmp = h_B;
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_zlantr( "F", lapack_uplo_const(opts.uplo),
                                                  lapack_diag_const(opts.diag),
                                             &h_Ak[s], &h_Ak[s], h_A_tmp, &h_lda[s], work );
                Bnorm[s] = lapackf77_zlange( "F", &h_M[s], &h_N[s], h_B_tmp, &h_ldb[s], work );
                h_A_tmp += h_Ak[s] * h_lda[s];
                h_B_tmp +=  h_N[s] * h_ldb[s];
            }
            
            // set A
            h_A_tmp = h_A;
            d_A_tmp = d_A;
            for (int i = 0; i < batchCount; i++) {
                magma_zsetmatrix( h_Ak[i], h_Ak[i], h_A_tmp, h_lda[i], d_A_tmp, h_ldda[i], opts.queue );
                h_A_tmp += h_Ak[i] * h_lda[i];
                d_A_tmp += h_Ak[i] * h_ldda[i];
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            // set B
            h_B_tmp = h_B;
            d_B_tmp = d_B;
            for (int i = 0; i < batchCount; i++) {
                magma_zsetmatrix( h_M[i],  h_N[i],  h_B_tmp, h_ldb[i], d_B_tmp, h_lddb[i], opts.queue );
                h_B_tmp += h_N[i] * h_ldb[i];
                d_B_tmp += h_N[i] * h_lddb[i];
            }
            
            magma_time = magma_sync_wtime( opts.queue );
            #if 0
            magmablas_ztrmm_vbatched_max_nocheck(
                    opts.side, opts.uplo, opts.transA, opts.diag,
                    d_M, d_N,
                    alpha, dA_array, d_ldda,
                           dB_array, d_lddb,
                    batchCount, max_M, max_N, opts.queue );
            #else
            magmablas_ztrmm_vbatched(
                    opts.side, opts.uplo, opts.transA, opts.diag,
                    d_M, d_N,
                    alpha, dA_array, d_ldda,
                           dB_array, d_lddb,
                    batchCount, opts.queue );
            #endif
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            
            h_B_tmp = h_Bmagma;
            for (int i = 0; i < batchCount; i++) {
                magma_zgetmatrix( h_M[i], h_N[i], hB_array[i], h_lddb[i], h_B_tmp, h_ldb[i], opts.queue );
                h_B_tmp += h_ldb[i] * h_N[i];
            }
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                hA_array[0] = h_A;
                hB_array[0] = h_B;
                for (int s = 1; s < batchCount; s++) {
                    hA_array[s] = hA_array[s-1] + h_lda[s-1] * h_Ak[s-1];
                    hB_array[s] = hB_array[s-1] + h_ldb[s-1] * h_N[s-1];
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int s=0; s < batchCount; s++){
                    blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                                   lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                                   &h_M[s], &h_N[s],
                                   &alpha, hA_array[s], &h_lda[s],
                                           hB_array[s], &h_ldb[s] );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dB - B| / (gamma_{k}|A||Bin|); k = Ak; no beta
                magma_error = 0;
                
                h_B_tmp = h_B;
                h_Bmagma_tmp = h_Bmagma;
                for (int s  = 0; s < batchCount; s++) {
                    normalize = sqrt(double(h_Ak[s]))*Anorm[s]*Bnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Bsize = h_ldb[s]*h_N[s];
                    blasf77_zaxpy( &Bsize, &c_neg_one, h_B_tmp, &ione, h_Bmagma_tmp, &ione );
                    error = lapackf77_zlange( "F", &h_M[s], &h_N[s], h_Bmagma_tmp, &h_ldb[s], work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                    
                    h_B_tmp      += h_ldb[s] * h_N[s];
                    h_Bmagma_tmp += h_ldb[s] * h_N[s];
                }
                bool okay = (magma_error < tol);
                status += ! okay;
                
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long)batchCount,
                       (long long)max_M, (long long)max_N,
                       magma_perf, 1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long)batchCount,
                       (long long)max_M, (long long)max_N,
                       magma_perf, 1000.*magma_time);
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_Bmagma );
            
            magma_free( d_A );
            magma_free( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_lddb );
    magma_free( dA_array );
    magma_free( dB_array );

    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_lddb );

    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );

    magma_free_cpu( hA_array );
    magma_free_cpu( hB_array );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
