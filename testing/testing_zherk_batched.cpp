/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Chongxiao Cao
       @author Tingxing Dong
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

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zherk_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf=0., cpu_time=0.;
    double          magma_error, Cnorm, work[1];
    magma_int_t i, N, K;
    magma_int_t Ak, An;
    magma_int_t sizeA, sizeC;
    magma_int_t lda, ldc, ldda, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t NN;
    magma_int_t batchCount;
 
    magmaDoubleComplex *h_A, *h_C, *h_Cmagma;
    magmaDoubleComplex *d_A, *d_C;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double alpha = 0.29;
    double beta  = -0.48;
    magmaDoubleComplex **A_array = NULL;
    magmaDoubleComplex **C_array = NULL;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    #ifdef COMPLEX
    if (opts.transA == MagmaTrans) {
        opts.transA = MagmaConjTrans; 
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #else
    if (opts.transA == MagmaConjTrans) {
        opts.transA = MagmaTrans; 
        printf("%% WARNING: transA = MagmaConjTrans changed to MagmaTrans\n");
    }
    #endif
   
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    
    printf("%% BatchCount   N     K   MAGMA Gflop/s (ms)    CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_ZHERK( K, N ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
            } else {
                lda = An = K;
                Ak = N;
            }

            ldc = N;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            NN = N * batchCount;

            sizeA = lda*Ak*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,  sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_C,  sizeC ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Cmagma,  sizeC  ));
            
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*Ak*batchCount ));
            TESTING_CHECK( magma_zmalloc( &d_C, lddc*N*batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &A_array, batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &C_array, batchCount * sizeof(magmaDoubleComplex*) ));

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            for (i=0; i < batchCount; i++)
            {
                magma_zmake_hpd( N, h_C + i * ldc * N, ldc ); // need modification
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( An, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_zset_pointer( A_array, d_A, lda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_zset_pointer( C_array, d_C, ldc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_zherk_batched(opts.uplo, opts.transA, N, K,
                             alpha, A_array, ldda,
                             beta,  C_array, lddc, batchCount, opts.queue);
                             
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetmatrix( N, NN, d_C, lddc, h_Cmagma, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                for (i=0; i < batchCount; i++)
                {
                    blasf77_zherk( lapack_uplo_const(opts.uplo),
                                   lapack_trans_const(opts.transA),
                                   &N, &K,
                                   &alpha, h_A + i*lda*Ak, &lda,
                                   &beta,  h_C + i*ldc*N, &ldc );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for magma, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                sizeC = ldc*N;
                magma_error = 0;
                for (i=0; i < batchCount; i++)
                {
                    blasf77_zaxpy( &sizeC, &c_neg_one, h_C+i*ldc*N, &ione, h_Cmagma+i*ldc*N, &ione );
                    Cnorm      = safe_lapackf77_zlanhe( "fro", lapack_uplo_const(opts.uplo), &N, h_C     +i*ldc*N, &ldc, work );
                    double err = safe_lapackf77_zlanhe( "fro", lapack_uplo_const(opts.uplo), &N, h_Cmagma+i*ldc*N, &ldc, work )
                               / Cnorm;
                    if ( isnan(err) || isinf(err) ) {
                        magma_error = err;
                        break;
                    }
                    magma_error = max( err, magma_error );
                }
                
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%10lld %5lld %5lld    %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) batchCount, (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%10lld %5lld %5lld    %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount, (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time);
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_C  );
            magma_free_cpu( h_Cmagma  );

            magma_free( d_A );
            magma_free( d_C );
            magma_free( A_array );
            magma_free( C_array );
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
