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
#include "magma_operators.h"  // for MAGMA_Z_DIV
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #endif

    #define hA(i_, j_) (hA + (i_) + (j_)*lda)

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf=0, magma_time=0, dev_perf, dev_time, cpu_perf=0, cpu_time=0;
    double          magma_error=0, dev_error, lapack_error, work[1];
    magma_int_t M, N, info;
    magma_int_t Ak;
    magma_int_t sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magmaDoubleComplex *hA, *hB, *hBdev, *hBmagma, *hBlapack, *hX;
    magmaDoubleComplex_ptr dA, dB;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.tolerance = 100;           // default
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );

    printf("%% side = %s, uplo = %s, transA = %s, diag = %s, ngpu = %lld\n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag), (long long) abs_ngpu);

    printf("%%   M     N  MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)      MAGMA     CUBLAS   LAPACK error\n");
    printf("%%============================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_ZTRSM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak  = M;
            } else {
                lda = N;
                Ak  = N;
            }

            ldb = M;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default

            //sizeA = lda*Ak;
            sizeB = ldb*N;

            TESTING_CHECK( magma_zmalloc_cpu( &hA,       lda*Ak  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hB,       ldb*N   ));
            TESTING_CHECK( magma_zmalloc_cpu( &hX,       ldb*N   ));
            TESTING_CHECK( magma_zmalloc_cpu( &hBlapack, ldb*N   ));
            TESTING_CHECK( magma_zmalloc_cpu( &hBdev,    ldb*N   ));
            TESTING_CHECK( magma_zmalloc_cpu( &hBmagma,  ldb*N   ));

            TESTING_CHECK( magma_zmalloc( &dA,       ldda*Ak ));
            TESTING_CHECK( magma_zmalloc( &dB,       lddb*N  ));

            /* Initialize the matrices */
            magma_generate_matrix( opts, Ak, Ak, hA, lda );

            // set unused data to nan
            magma_int_t Ak_1 = Ak - 1;
            if (opts.uplo == MagmaLower)
                lapackf77_zlaset( "upper", &Ak_1, &Ak_1, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &hA[ 1*lda ], &lda );
            else
                lapackf77_zlaset( "lower", &Ak_1, &Ak_1, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &hA[ 1     ], &lda );

            // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
            // If diag == Unit, the diagonal is replaced; this is still well-conditioned.
            // First, brute force positive definiteness.
            for (int i = 0; i < Ak; ++i) {
                hA[ i + i*lda ] += Ak;
            }
            lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &Ak, hA, &lda, &info );
            assert( info == 0 );

            lapackf77_zlarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, hB, &ldb, hBlapack, &lda );
            magma_zsetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );

            /* =====================================================================
               Performs operation using MAGMABLAS (only with CUDA)
               =================================================================== */
            #if defined(HAVE_CUBLAS)
                magma_zsetmatrix( M, N, hB, ldb, dB(0,0), lddb, opts.queue );

                magma_time = magma_sync_wtime( opts.queue );
                if (opts.ngpu == 1) {
                    magmablas_ztrsm( opts.side, opts.uplo, opts.transA, opts.diag,
                                     M, N,
                                     alpha, dA(0,0), ldda,
                                            dB(0,0), lddb, opts.queue );
                }
                else {
                    magma_ztrsm_m( abs_ngpu, opts.side, opts.uplo, opts.transA, opts.diag,
                                   M, N,
                                   alpha, dA(0,0), ldda,
                                          dB(0,0), lddb );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;

                magma_zgetmatrix( M, N, dB(0,0), lddb, hBmagma, ldb, opts.queue );
            #endif

            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_zsetmatrix( M, N, hB, ldb, dB(0,0), lddb, opts.queue );

            dev_time = magma_sync_wtime( opts.queue );
            magma_ztrsm( opts.side, opts.uplo, opts.transA, opts.diag,
                         M, N,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;

            magma_zgetmatrix( M, N, dB(0,0), lddb, hBdev, ldb, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrsm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &alpha, hA, &lda,
                                       hBlapack, &ldb );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - 1/alpha*A*x|| / (||A||*||x||)
            magmaDoubleComplex inv_alpha = MAGMA_Z_DIV( c_one, alpha );
            double normR, normX, normA;
            normA = lapackf77_zlantr( "M",
                                      lapack_uplo_const(opts.uplo),
                                      lapack_diag_const(opts.diag),
                                      &Ak, &Ak, hA, &lda, work );

            #if defined(HAVE_CUBLAS)
                // check magma
                memcpy( hX, hBmagma, sizeB*sizeof(magmaDoubleComplex) );
                blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &inv_alpha, hA, &lda,
                                           hX, &ldb );

                blasf77_zaxpy( &sizeB, &c_neg_one, hB, &ione, hX, &ione );
                normR = lapackf77_zlange( "M", &M, &N, hX,      &ldb, work );
                normX = lapackf77_zlange( "M", &M, &N, hBmagma, &ldb, work );
                magma_error = normR/(normX*normA);
            #endif

            // check cuBLAS / clBLAS
            memcpy( hX, hBdev, sizeB*sizeof(magmaDoubleComplex) );
            blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &M, &N,
                           &inv_alpha, hA, &lda,
                                       hX, &ldb );

            blasf77_zaxpy( &sizeB, &c_neg_one, hB, &ione, hX, &ione );
            normR = lapackf77_zlange( "M", &M, &N, hX,    &ldb, work );
            normX = lapackf77_zlange( "M", &M, &N, hBdev, &ldb, work );
            dev_error = normR/(normX*normA);

            bool okay = (magma_error < tol && dev_error < tol);
            status += ! okay;
            if ( opts.lapack ) {
                // check lapack
                // this verifies that the matrix wasn't so bad that it couldn't be solved accurately.
                memcpy( hX, hBlapack, sizeB*sizeof(magmaDoubleComplex) );
                blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &inv_alpha, hA, &lda,
                                           hX, &ldb );

                blasf77_zaxpy( &sizeB, &c_neg_one, hB, &ione, hX, &ione );
                normR = lapackf77_zlange( "M", &M, &N, hX,       &ldb, work );
                normX = lapackf77_zlange( "M", &M, &N, hBlapack, &ldb, work );
                lapack_error = normR/(normX*normA);

                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                        (long long) M, (long long) N,
                        magma_perf,  1000.*magma_time,
                        dev_perf,    1000.*dev_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, dev_error, lapack_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )   %8.2e   %8.2e     ---      %s\n",
                        (long long) M, (long long) N,
                        magma_perf,  1000.*magma_time,
                        dev_perf,    1000.*dev_time,
                        magma_error, dev_error,
                        (okay ? "ok" : "failed"));
            }

            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hX );
            magma_free_cpu( hBlapack );
            magma_free_cpu( hBdev    );
            magma_free_cpu( hBmagma  );

            magma_free( dA );
            magma_free( dB );
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
