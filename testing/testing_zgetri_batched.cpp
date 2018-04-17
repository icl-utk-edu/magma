/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Mark Gates

   @precisions normal z -> s d c
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
#include "../control/magma_threadsetting.h"  // internal header
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetri_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    // constants
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_Ainv, *h_R, *work, unused[1];
    magmaDoubleComplex_ptr d_A, d_invA;
    magmaDoubleComplex_ptr *dA_array;
    magmaDoubleComplex_ptr *dinvA_array;
    magma_int_t **dipiv_array;
    magma_int_t *dinfo_array;
    magma_int_t *ipiv, *cpu_info, iunused[1];
    magma_int_t *d_ipiv, *d_info;
    magma_int_t N, n2, lda, ldda, info, info1, info2, lwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magmaDoubleComplex tmp;
    double  error, rwork[1];
    magma_int_t columns;
    int status = 0;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    
    magma_int_t batchCount = opts.batchcount;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% batchCount   N    CPU Gflop/s (ms)    GPU Gflop/s (ms)   ||I - A*A^{-1}||_1 / (N*cond(A))\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {    
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N * batchCount;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            // This is the correct flops but since this getri_batched is based on
            // 2 trsm = getrs and to know the real flops I am using the getrs one
            //gflops = (FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRI( N ))/ 1e9 * batchCount;
            gflops = (FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRS( N, N ))/ 1e9 * batchCount;

            // query for workspace size
            lwork = -1;
            lapackf77_zgetri( &N, unused, &lda, iunused, &tmp, &lwork, &info );
            if (info != 0) {
                printf("lapackf77_zgetri returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            lwork = magma_int_t( MAGMA_Z_REAL( tmp ));
            
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv,     N * batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &work,     lwork*batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,      n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Ainv,   n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_R,      n2     ));
            
            TESTING_CHECK( magma_zmalloc( &d_A,      ldda*N * batchCount ));
            TESTING_CHECK( magma_zmalloc( &d_invA,   ldda*N * batchCount ));
            TESTING_CHECK( magma_imalloc( &d_ipiv,   N * batchCount ));
            TESTING_CHECK( magma_imalloc( &d_info,   batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dinvA_array, batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));
            TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            columns = N * batchCount;
            lapackf77_zlacpy( MagmaFullStr, &N, &columns, h_A, &lda, h_R,  &lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &columns, h_A, &lda, h_Ainv, &lda );
            magma_zsetmatrix( N, columns, h_R, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( dA_array, d_A, ldda, 0, 0, ldda * N, batchCount, opts.queue );
            magma_zset_pointer( dinvA_array, d_invA, ldda, 0, 0, ldda * N, batchCount, opts.queue );
            magma_iset_pointer( dipiv_array, d_ipiv, 1, 0, 0, N, batchCount, opts.queue );

            gpu_time = magma_sync_wtime( opts.queue );
            info1 = magma_zgetrf_batched( N, N, dA_array, ldda, dipiv_array, dinfo_array, batchCount, opts.queue);
            info2 = magma_zgetri_outofplace_batched( N, dA_array, ldda, dipiv_array, dinvA_array, ldda, dinfo_array, batchCount, opts.queue);
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;

            // check correctness of results through "dinfo_magma" and correctness of argument through "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, opts.queue );
            for (magma_int_t i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgetrf_batched matrix %lld returned error %lld\n", (long long) i, (long long) cpu_info[i] );
                }
            }
            if (info1 != 0) printf("magma_zgetrf_batched returned argument error %lld: %s.\n", (long long) info1, magma_strerror( info1 ));
            if (info2 != 0) printf("magma_zgetri_batched returned argument error %lld: %s.\n", (long long) info2, magma_strerror( info2 ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                    magma_int_t locinfo;
                    lapackf77_zgetrf(&N, &N, h_Ainv + i*lda*N, &lda, ipiv + i*N, &locinfo);
                    if (locinfo != 0) {
                        printf("lapackf77_zgetrf returned error %lld: %s.\n",
                               (long long) locinfo, magma_strerror( locinfo ));
                    }
                    lapackf77_zgetri(&N, h_Ainv + i*lda*N, &lda, ipiv + i*N, work + i*lwork, &lwork, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zgetri returned error %lld: %s.\n",
                               (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                
                printf("%10lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) batchCount, (long long) N, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000. );
            }
            else {
                printf("%10lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) batchCount, (long long) N, gpu_perf, gpu_time*1000. );
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                magma_igetvector( N*batchCount, d_ipiv, 1, ipiv, 1, opts.queue );
                magma_zgetmatrix( N, N*batchCount, d_invA, ldda, h_Ainv, lda, opts.queue );
                error = 0;
                for (magma_int_t i=0; i < batchCount; i++)
                {
                    for (magma_int_t k=0; k < N; k++) {
                        if (ipiv[i*N+k] < 1 || ipiv[i*N+k] > N )
                        {
                            printf("error for matrix %lld ipiv @ %lld = %lld\n",
                                    (long long) i, (long long) k, (long long) ipiv[i*N+k] );
                            error = -1;
                        }
                    }
                    if (error == -1) {
                        break;
                    }
                    
                    // compute 1-norm condition number estimate, following LAPACK's zget03
                    double normA, normAinv, rcond, err;
                    normA    = lapackf77_zlange( "1", &N, &N, h_A    + i*lda*N, &lda, rwork );
                    normAinv = lapackf77_zlange( "1", &N, &N, h_Ainv + i*lda*N, &lda, rwork );
                    if ( normA <= 0 || normAinv <= 0 ) {
                        rcond = 0;
                        err = 1 / (tol/opts.tolerance);  // == 1/eps
                    }
                    else {
                        rcond = (1 / normA) / normAinv;
                        // R = I
                        // R -= A*A^{-1}
                        // err = ||I - A*A^{-1}|| / ( N ||A||*||A^{-1}|| ) = ||R|| * rcond / N, using 1-norm
                        lapackf77_zlaset( "full", &N, &N, &c_zero, &c_one, h_R + i*lda*N, &lda );
                        blasf77_zgemm( "no", "no", &N, &N, &N, &c_neg_one,
                                       h_A    + i*lda*N, &lda,
                                       h_Ainv + i*lda*N, &lda, &c_one,
                                       h_R    + i*lda*N, &lda );
                        err = lapackf77_zlange( "1", &N, &N, h_R + i*lda*N, &lda, rwork );
                        err = err * rcond / N;
                    }
                    if (std::isnan(err) || std::isinf(err)) {
                        error = err;
                        break;
                    }
                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("\n");
            }

            magma_free_cpu( cpu_info );
            magma_free_cpu( ipiv   );
            magma_free_cpu( work   );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_Ainv );
            magma_free_cpu( h_R    );
            
            magma_free( d_A );
            magma_free( d_invA );
            magma_free( d_ipiv );
            magma_free( d_info );
            
            magma_free( dA_array );
            magma_free( dinvA_array );
            magma_free( dinfo_array );
            magma_free( dipiv_array );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
