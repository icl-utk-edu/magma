/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    double          magma_error, dev_error, normalize, work[1];
    magma_int_t M, N;
    magma_int_t An;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magmaDoubleComplex *hA, *hB, *hC, *hCmagma, *hCdev;
    magmaDoubleComplex *dA, *dB, *dC;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double Anorm, Bnorm, Cnorm;

    // See testing_zgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;

    #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_SYCL)
    // for CUDA, we can check MAGMA vs. CUBLAS, without running LAPACK
    printf("%% If running lapack (option --lapack), MAGMA and %s errors are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
            g_platform_str, g_platform_str );
        
    printf("%% side = %s, uplo = %s\n", 
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo));

    printf("%%   M     N   MAGMA Gflop/s (ms)  %s Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   %s error\n", 
           g_platform_str, g_platform_str);
   #else
    // for others, we need LAPACK for check
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    printf("%% side = %s, uplo = %s\n", 
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo));
    printf("%%   M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error\n");
    #endif
    printf("%%===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_ZHEMM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                An = M;
            } else {
                lda = N;
                An = N;
            }
            ldb = ldc = M;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An;
            sizeB = ldb*N;
            sizeC = ldc*N;

            TESTING_CHECK( magma_zmalloc_cpu(&hA, sizeA) );
            TESTING_CHECK( magma_zmalloc_cpu(&hB, sizeB) );
            TESTING_CHECK( magma_zmalloc_cpu(&hC, sizeC) );
            TESTING_CHECK( magma_zmalloc_cpu(&hCmagma, sizeC) );
            TESTING_CHECK( magma_zmalloc_cpu(&hCdev, sizeC) );

            TESTING_CHECK( magma_zmalloc(&dA, ldda*An) );
            TESTING_CHECK( magma_zmalloc(&dB, lddb*N ) );
            TESTING_CHECK( magma_zmalloc(&dC, lddc*N ) );

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, hC );

            Anorm = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &An, hA, &lda, work );
            Bnorm = lapackf77_zlange( "F", &M, &N, hB, &ldb, work );
            Cnorm = lapackf77_zlange( "F", &M, &N, hC, &ldc, work );

            magma_zsetmatrix( An, An, hA, lda, dA, ldda, opts.queue );
            magma_zsetmatrix( M, N, hB, ldb, dB, lddb, opts.queue );

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( M, N, hC, ldc, dC, lddc, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_zhemm( 
                    opts.side, opts.uplo, M, N, 
                    alpha, dA, ldda, 
                           dB, lddb, 
                    beta,  dC, lddc, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_zgetmatrix( M, N, dC, lddc, hCmagma, ldc, opts.queue );

            /* =====================================================================
               Performs operation using device BLAS (if available)
               =================================================================== */
            #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_SYCL)
            magma_zsetmatrix( M, N, hC, ldc, dC, lddc, opts.queue );

            dev_time = magma_sync_wtime( opts.queue );
            magma_zhemm( 
                opts.side, opts.uplo, M, N, 
                alpha, dA, ldda, 
                       dB, lddb, 
                beta,  dC, lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            magma_zgetmatrix( M, N, dC, lddc, hCdev, ldc, opts.queue );
            #endif

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_zhemm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo), 
                    &M, &N, &alpha, hA, &lda, hB, &ldb, &beta, hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared to lapack
                // error = |dC - C| / (gamma_{k+2}|A||B| + gamma_2|Cin|); k = Am
                blasf77_zaxpy( &sizeC, &c_neg_one, hC, &ione, hCmagma, &ione );
                normalize = sqrt(double(An+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm;
                if (normalize == 0) normalize = 1;
                magma_error = lapackf77_zlange( "F", &M, &N, hCmagma, &ldc, work ) / normalize;

                #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_SYCL)
                blasf77_zaxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = lapackf77_zlange( "F", &M, &N, hCdev, &ldc, work ) / normalize;

                bool okay = (magma_error < tol && dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %7.2f (%7.2f)   %8.2e      %8.2e   %s\n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, dev_error, (okay ? "ok" : "failed"));
                #else
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
                #endif
            }
            else {
                #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_SYCL)
                // compute MAGMABLAS error relative to CUBLAS
                blasf77_zaxpy( &sizeC, &c_neg_one, hCdev, &ione, hCmagma, &ione );
                normalize = sqrt(double(An+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm;
                if (normalize == 0) normalize = 1;
                magma_error = lapackf77_zlange( "F", &M, &N, hCmagma, &ldc, work ) / normalize;
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )    %8.2e         ---      %s\n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       magma_error, (okay ? "ok" : "failed"));
                #else
                printf("%5lld %5lld   %7.2f (%7.2f)   ---   (  ---  )     ---  \n",
                       (long long) M, (long long) N,
                       magma_perf, 1000.*magma_time);
                #endif
            }

            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCmagma );
            magma_free_cpu( hCdev );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
