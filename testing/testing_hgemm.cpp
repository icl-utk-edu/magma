/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Mark Gates
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
   -- Testing sgemm
*/
int main( int argc, char** argv)
{

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time;
    float          dev_error, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn, Wm, Wn, info;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc, lddw;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    float *hA, *hB, *hC, *hCdev;
    float *dW;
    magmaHalf_ptr dA, dB, dC;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    magmaHalf h_alpha = (magmaHalf)(alpha);
    magmaHalf h_beta  = (magmaHalf)(beta);
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // Allow 3*eps; real needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    // For half precision, there is no lapackf77_hlamch, please visit: 
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float eps = (float)(0.00097656);
    float tol = 3*eps;

    
    printf("%% If running with option --lapack (-l) or with checking (-c), GPU error is computed\n"
            "%% relative to CPU BLAS result in single precision.\n\n");
    printf("%% transA = %s, transB = %s\n",
            lapack_trans_const(opts.transA),
            lapack_trans_const(opts.transB) );
    printf("%%   M     N     K   GPU Gflop/s (ms)  GPU error\n");
    printf("%%========================================================================================================\n");

    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            } else {
                lda = Am = K;
                An = M;
            }
            
            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;
            
            // workspace for conversion between single anf half precisions
            Wm    = max(Am, Bm);
            Wn    = max(An, Bn);

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            lddw = magma_roundup(  Wm, opts.align );  // multiple of 32 by default
            
            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_smalloc_cpu( &hA,       lda*An ));
            TESTING_CHECK( magma_smalloc_cpu( &hB,       ldb*Bn ));
            TESTING_CHECK( magma_smalloc_cpu( &hC,       ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCdev,    ldc*N  ));
            
            TESTING_CHECK( magma_smalloc( &dW, lddw*Wn  ));
            TESTING_CHECK( magma_malloc( (void**)&dA, ldda*An*sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**)&dB, lddb*Bn*sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**)&dC, lddc*N *sizeof(magmaHalf)  ));

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_slarnv( &ione, ISEED, &sizeC, hC );

            magma_ssetmatrix( Am, An, hA, lda,  dW, lddw, opts.queue );
            magmablas_slag2h( Am, An, dW, lddw, dA, ldda, &info, opts.queue);
            if(info == 0) {
                printf("error in magmablas_slag2h( dA )\n");
            }

            magma_ssetmatrix( Bm, Bn, hB, ldb,  dW, lddw, opts.queue );
            magmablas_slag2h( Bm, Bn, dW, lddw, dB, lddb, &info, opts.queue);
            if(info == 0) {
                printf("error in magmablas_slag2h( dB )\n");
            }

            // for error checks
            float Anorm = lapackf77_slange( "F", &Am, &An, hA, &lda, work );
            float Bnorm = lapackf77_slange( "F", &Bm, &Bn, hB, &ldb, work );
            float Cnorm = lapackf77_slange( "F", &M,  &N,  hC, &ldc, work );
            
            /* =====================================================================
               Performs operation using GPU
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_ssetmatrix( M, N, hC, ldc, dW, lddw, opts.queue );
                magmablas_slag2h( M, N, dW, lddw, dC, lddc, &info, opts.queue);
                if(info == 0) {
                    printf("error in magmablas_slag2h( dC )\n");
                }

                magma_flush_cache( opts.cache );
                dev_time = magma_sync_wtime( opts.queue );
                magma_hgemm( opts.transA, opts.transB, M, N, K,
                             h_alpha, dA, ldda,
                                      dB, lddb,
                             h_beta,  dC, lddc,
                             opts.queue );
                dev_time = magma_sync_wtime( opts.queue ) - dev_time;
                dev_perf = gflops / dev_time;
                
                magmablas_hlag2s( M, N, dC, lddc, dW, lddw, opts.queue );
                magma_sgetmatrix( M, N, dW, lddw, hCdev, ldc, opts.queue );
            #else
                dev_time = 0.0;
                dev_perf = 0.0;
            #endif
            
            
            /* =====================================================================
               Check the result
               =================================================================== */
            #ifdef HAVE_CUBLAS
            if ( opts.lapack || opts.check ) {
                /* =====================================================================
                   Performs operation using CPU BLAS
                   =================================================================== */

                blasf77_sgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                
                // Compute forward error bound (see Higham, 2002, sec. 3.5),
                // modified to include alpha, beta, and input C.
                // ||R_magma - R_ref||_p / (gamma_{K+2} |alpha| ||A||_p ||B||_p + 2 |beta| ||C||_p ) < eps/2.
                // This should work with p = 1, inf, fro, but numerical tests
                // show p = 1, inf are very spiky and sometimes exceed eps.
                // We use gamma_n = sqrt(n)*u instead of n*u/(1-n*u), since the
                // former accurately represents statistical average rounding.
                // We allow a slightly looser tolerance.
                
                // use LAPACK for R_ref
                blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = lapackf77_slange( "F", &M, &N, hCdev, &ldc, work )
                            / (sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                
                bool okay = (dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld %5lld   %7.2f (%7.2f)    %8.2e   %s\n",
                        (long long) M, (long long) N, (long long) K,
                        dev_perf,    1000.*dev_time,
                        dev_error,
                        (okay ? "ok" : "failed"));
            }
            else {
            #else
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )       ---\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time );
            #endif

            #ifdef HAVE_CUBLAS
            }
            #endif
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCdev    );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            magma_free( dW );
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
