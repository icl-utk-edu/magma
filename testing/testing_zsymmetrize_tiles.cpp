/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zsymmetrize
   Code is very similar to testing_ztranspose.cpp
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_A;
    magma_int_t i, j, N, nb, size, lda, ldda, mstride, nstride, ntile, tile, offset;
    magma_int_t ione     = 1;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    nb = (opts.nb == 0 ? 64 : opts.nb);
    mstride = 2*nb;
    nstride = 3*nb;
    
    printf("%% uplo = %s, nb = %lld, mstride = %lld, nstride = %lld\n",
            lapack_uplo_const(opts.uplo), (long long) nb, (long long) mstride, (long long) nstride );
    printf("%%   N ntile   CPU GByte/s (ms)    GPU GByte/s (ms)    check\n");
    printf("%%==========================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            size   = lda*N;
            
            if ( N < nb ) {
                ntile = 0;
            } else {
                ntile = min( (N - nb)/mstride + 1,
                             (N - nb)/nstride + 1 );
            }
            // load each tile, save each tile
            gbytes = sizeof(magmaDoubleComplex) * 2.*nb*nb*ntile / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, size   ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_R, size   ));
            
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N ));
            
            /* Initialize the matrix */
            for( j = 0; j < N; ++j ) {
                for( i = 0; i < N; ++i ) {
                    h_A[i + j*lda] = MAGMA_Z_MAKE( i + j/10000., j );
                }
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_zsymmetrize_tiles( opts.uplo, nb, d_A, ldda, ntile, mstride, nstride, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation using naive in-place algorithm
               (LAPACK doesn't implement symmetrize)
               =================================================================== */
            cpu_time = magma_wtime();
            for( tile = 0; tile < ntile; ++tile ) {
                offset = tile*mstride + tile*nstride*lda;
                for( j = 0; j < nb; ++j ) {
                    for( i = 0; i < j; ++i ) {
                        if ( opts.uplo == MagmaLower ) {
                            h_A[offset + i + j*lda] = MAGMA_Z_CONJ( h_A[offset + j + i*lda] );
                        }
                        else {
                            h_A[offset + j + i*lda] = MAGMA_Z_CONJ( h_A[offset + i + j*lda] );
                        }
                    }
                    // real diagonal
                    h_A[offset + j + j*lda] = MAGMA_Z_MAKE( MAGMA_Z_REAL( h_A[offset + j + j*lda] ), 0 );
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
            
            blasf77_zaxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_zlange("f", &N, &N, h_R, &lda, work);

            printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   (long long) N, (long long) ntile,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (error == 0. ? "ok" : "failed") );
            status += ! (error == 0.);
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_R );
            
            magma_free( d_A );
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
