/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Mark gates
   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
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
   -- Testing zgesv_batched
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time = 0;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t *dipiv;
    magma_int_t *ipiv;
    magma_int_t N, Nband, KL, KU, KV, nrhs, lda, ldb, ldda, lddb, info = 0, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs       = opts.nrhs;
    KL         = opts.kl;
    KU         = opts.ku;
    KV         = KL + KU;

    printf("%% ## INFO ##: Gflop/s calculation is not available\n");
    printf("%% Lower bandwidth (KL) = %lld\n", (long long)KL);
    printf("%% Upper bandwidth (KU) = %lld\n", (long long)KU);
    printf("%% N  NRHS   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            Nband  = KL + 1 + KV; // need extra KL superdiagonals for the upper factor
            lda    = Nband;
            ldb    = N;
            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( ldb, opts.align );
            gflops = 0.;  // TODO: gflop formula for gbsv?

            sizeA = lda*N;
            sizeB = ldb*nrhs;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B, sizeB ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, sizeB ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));

            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N    ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs ));
            TESTING_CHECK( magma_imalloc( &dipiv, N ));

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );

            // random initialization of h_A seems to produce
            // some matrices that are singular, the additive statements below
            // seem to avoid that
            #pragma omp parallel for schedule(dynamic)
            for(int j = 0; j < lda*N; j++) {
                #if defined(PRECISION_c) || defined(PRECISION_z)
                h_A[j]= MAGMA_Z_MAKE(MAGMA_Z_REAL(h_A[j]) + 20., MAGMA_Z_IMAG(h_A[j]) + 20.);
                #else
		h_A[j] += 20.;
                #endif
            }


            magma_zsetmatrix( Nband, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N,     nrhs, h_B, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */

            if(opts.version == 1) {
                // sync. interface
                gpu_time = magma_wtime();
                magma_zgbsv_native(
                    N, KL, KU, nrhs,
                    d_A, ldda, dipiv,
                    d_B, lddb, &info);
                gpu_time = magma_wtime() - gpu_time;
            }
            else {
                // async. interface

                // query wrokspace
                magma_int_t lwork[1] = {-1};
                magma_zgbsv_native_work(
                    N, KL, KU, nrhs,
                    NULL, ldda, NULL,
                    NULL, lddb, &info, NULL, lwork, opts.queue);

                void* device_work = NULL;
                TESTING_CHECK( magma_malloc(&device_work, lwork[0]) );

                // time async. call only
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zgbsv_native_work(
                    N, KL, KU, nrhs,
                    d_A, ldda, dipiv,
                    d_B, lddb, &info, device_work, lwork, opts.queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                magma_free( device_work );
            }
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("magma_zgbsv_gpu returned error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_B, lddb, h_X, ldb, opts.queue );

            error = 0;
            Anorm = lapackf77_zlangb("I", &N, &KL, &KU, h_A + KL, &lda, work);
            Xnorm = lapackf77_zlange("I", &N, &nrhs, h_X, &ldb, work);

            for(magma_int_t j = 0; j < nrhs; j++) {
                blasf77_zgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
                               &c_one, h_A + KL     , &lda,
                                       h_X + j * ldb, &ione,
                           &c_neg_one, h_B + j * ldb, &ione);
            }

            Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B, &ldb, work);

            error = Rnorm/(N*Anorm*Xnorm);
            bool okay = (error < tol);
            status += ! okay;

            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {

                cpu_time = magma_wtime();
                lapackf77_zgbsv(&N, &KL, &KU, &nrhs, h_A, &lda, ipiv, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;

                if (info != 0) {
                    printf("lapackf77_zgesv returned error %lld: %s.\n",
                            (long long)info, magma_strerror( info ));
                }
                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs,
                        cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs,
                        gpu_perf, 1000.*gpu_time,
                        error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );

            magma_free( d_A );
            magma_free( d_B );
            magma_free( dipiv );

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
