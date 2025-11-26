/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Ahmad Abdelfattah
       @precisions normal z -> c d s
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zunmqr_gpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double Cnorm, error, magma_error, work[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t Am, m, n, k, nn, kk, size, info, batchCount;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t ldc, lddc, lda, ldda, lwork;
    magmaDoubleComplex *hC, *hR, *hA, *hwork, *htau;
    magmaDoubleComplex_ptr dC, dA, dtau;
    magmaDoubleComplex **dC_array, **dA_array, **dtau_array;
    magma_int_t *hinfo_array, *dinfo_array;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;

    // need slightly looser bound (60*eps instead of 30*eps) for some tests
    opts.tolerance = max( 60., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");

    opts.lapack|= opts.check;

    magma_side_t  side  = opts.side;
    magma_trans_t trans = opts.transA;

    #ifdef COMPLEX
    if(trans == MagmaTrans) {
        trans = MagmaConjTrans;
        printf("%% WARNING: trans = MagmaTrans is not a valid option for xUNMQR, changed to MagmaConjTrans\n\n");
    }
    #else
    if(trans == MagmaConjTrans) {
        trans = MagmaTrans;
        printf("%% WARNING: trans = MagmaConjTrans is not a valid option for xORMQR, changed to MagmaTrans\n\n");
    }
    #endif

    TESTING_CHECK( magma_malloc( (void**) &dA_array,   batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &dC_array,   batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &dtau_array, batchCount * sizeof(magmaDoubleComplex*) ));

    TESTING_CHECK( magma_imalloc(&dinfo_array,     batchCount * sizeof(magma_int_t) ));
    TESTING_CHECK( magma_imalloc_cpu(&hinfo_array, batchCount * sizeof(magma_int_t) ));

    printf("%% side = %c, trans = %c \n", lapacke_side_const( side ), lapacke_trans_const( trans ) );
    printf("%% BatchCount     M     N     K   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||R||_F / ||QC||_F\n");
    printf("%%==============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m  = opts.msize[itest];
            n  = opts.nsize[itest];
            k  = opts.ksize[itest];
            nn = n*batchCount;
            kk = k*batchCount;

            // A is m x k (left) or n x k (right)
            Am     = (side == MagmaLeft ? m : n);

            ldc  = m;
            lda  = Am;
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZUNMQR( m, n, k, side ) * batchCount / 1e9;

            if ( side == MagmaLeft && m < k ) {
                printf( "%5lld %5lld %5lld   %4c   %5c   skipping because side=left  and m < k\n",
                        (long long) m, (long long) n, (long long) k,
                        lapacke_side_const( side ),
                        lapacke_trans_const( trans ) );
                continue;
            }
            if ( side == MagmaRight && n < k ) {
                printf( "%5lld %5lld %5lld   %4c   %5c   skipping because side=right and n < k\n",
                        (long long) m, (long long) n, (long long) k,
                        lapacke_side_const( side ),
                        lapacke_trans_const( trans ) );
                continue;
            }

            TESTING_CHECK( magma_zmalloc_cpu( &hC,     ldc*n*batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &hR,     ldc*n*batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &hA,     lda*k*batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &htau,   k*batchCount ));

            TESTING_CHECK( magma_zmalloc( &dC,  lddc*n*batchCount ));
            TESTING_CHECK( magma_zmalloc( &dA,  ldda*k*batchCount ));
            TESTING_CHECK( magma_zmalloc( &dtau, k*batchCount ));

            // C is full, m x n
            size = ldc*nn;
            lapackf77_zlarnv( &ione, ISEED, &size, hC );
            magma_zsetmatrix( m, nn, hC, ldc, dC, lddc, opts.queue );

            // query workspace for geqrf
            magmaDoubleComplex htmp[1];
            magma_int_t lwork_tmp = -1;
            lapackf77_zgeqrf( &Am, &k, hA, &lda, htau, htmp, &lwork_tmp, &info );
            lwork = (magma_int_t)MAGMA_Z_REAL( htmp[0] );
            // query workspace for unmqr
            lwork_tmp = -1;
            lapackf77_zunmqr( lapack_side_const( side ), lapack_trans_const( trans ),
                              &m, &n, &k, hA, &lda, htau, hC, &ldc, htmp, &lwork_tmp, &info );
            // take the max workspace size across qr and unmqr
            lwork = max(lwork, (magma_int_t)MAGMA_Z_REAL( htmp[0] ));
            TESTING_CHECK( magma_zmalloc_cpu( &hwork, lwork*batchCount ));

            // generate A (Am x k) and then
            // compute QR factorization to get Householder vectors in A & tau
            #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
            magma_int_t nthreads = magma_get_lapack_numthreads();
            magma_set_lapack_numthreads(1);
            magma_set_omp_numthreads(nthreads);
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(magma_int_t s = 0; s < batchCount; s++) {
                magma_int_t locinfo;
                magma_generate_matrix( opts, Am, k, hA + s*lda*k, lda );
                lapackf77_zgeqrf( &Am, &k, hA + s*lda*k, &lda, htau + s * k,
                                  hwork + s*lwork, &lwork, &locinfo );
                if (locinfo != 0) {
                    printf("lapackf77_zgeqrf returned error %lld: %s.\n",
                       (long long) locinfo, magma_strerror( locinfo ));
                }
            }
            #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
            magma_set_lapack_numthreads(nthreads);
            #endif

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( Am, kk, hA, lda, dA, ldda, opts.queue );
            magma_zsetmatrix( m,  nn, hC, ldc, dC, lddc, opts.queue );
            magma_zsetvector( batchCount*k, htau, 1, dtau, 1, opts.queue );

            magma_zset_pointer( dA_array, dA, ldda,  0, 0, ldda*k, batchCount, opts.queue );
            magma_zset_pointer( dC_array, dC, lddc,  0, 0, lddc*n, batchCount, opts.queue );
            magma_zset_pointer( dtau_array, dtau, k, 0, 0,      k, batchCount, opts.queue );

            // query workspace
            int64_t lwork_device[1] = {-1};
            void* device_work = NULL;
            magma_zunmqr_batched(
                side, trans, m, n, k,
                dA_array, ldda, dtau_array,
                dC_array, lddc,
                device_work, lwork_device,
                dinfo_array, batchCount, opts.queue);

            // allocate workspace
            if(lwork_device[0] > 0) {
                magma_malloc(&device_work, lwork_device[0]);
            }

            // time actual call
            gpu_time = magma_sync_wtime( opts.queue );
            magma_zunmqr_batched(
                side, trans, m, n, k,
                dA_array, ldda, dtau_array,
                dC_array, lddc,
                device_work, lwork_device,
                dinfo_array, batchCount, opts.queue);
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

            if(device_work != NULL) {
                magma_free(device_work);
            }
            gpu_perf = gflops / gpu_time;

            // check info
            magma_igetvector(batchCount, dinfo_array, 1, hinfo_array, 1, opts.queue);
            for(magma_int_t s = 0; s < batchCount; s++) {
                if(hinfo_array[s] != 0) {
                    printf("magma_zunmqr_batched matrix %lld returned internal error %lld\n",
                            (long long) s, (long long) hinfo_array[s] );
                }
            }

            magma_zgetmatrix( m, nn, dC, lddc, hR, ldc, opts.queue );

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if(opts.lapack) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for(magma_int_t s = 0; s < batchCount; s++) {
                    magma_int_t locinfo;
                    lapackf77_zunmqr( lapack_side_const( side ), lapack_trans_const( trans ),
                                  &m, &n, &k,
                                  hA + s*lda*k, &lda, htau+s*k, hC+s*ldc*n, &ldc, hwork + s*lwork, &lwork, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zunmqr returned error %lld: %s.\n",
                           (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               compute relative error |QC_magma - QC_lapack| / |QC_lapack|
               =================================================================== */
            if(opts.check) {
                size = ldc*n;
                magma_error = 0;
                for(magma_int_t s = 0; s < batchCount; s++) {
                    blasf77_zaxpy( &size, &c_neg_one, hC+s*ldc*n, &ione, hR+s*ldc*n, &ione );
                    Cnorm = lapackf77_zlange( "Fro", &m, &n, hC+s*ldc*n, &ldc, work );
                    error = lapackf77_zlange( "Fro", &m, &n, hR+s*ldc*n, &ldc, work ) / (magma_dsqrt(m*n) * Cnorm);
                    magma_error = magma_max_nan( error, magma_error );
                }
            }

            if(opts.lapack) {
                printf( "  %10lld %5lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   ",
                        (long long) batchCount, (long long) m, (long long) n, (long long) k,
                        cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.);
            }
            else {
                printf( "  %10lld %5lld %5lld %5lld   ---     (  ---  )   %7.2f (%7.2f)   ",
                        (long long) batchCount, (long long) m, (long long) n, (long long) k,
                        gpu_perf, gpu_time*1000. );
            }

            if(opts.check) {
                printf( "%8.2e   %s", magma_error, (magma_error < tol ? "ok" : "failed") );
                status += ! (magma_error < tol);
            }
            else {
                printf("  ---   ");
            }
            printf("\n");

            magma_free_cpu( hC );
            magma_free_cpu( hR );
            magma_free_cpu( hA );
            magma_free_cpu( hwork );
            magma_free_cpu( htau );

            magma_free( dC );
            magma_free( dA );
            magma_free( dtau );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free(dA_array);
    magma_free(dC_array);
    magma_free(dtau_array);
    magma_free(dinfo_array);

    magma_free_cpu(hinfo_array);

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
